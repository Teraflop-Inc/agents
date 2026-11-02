# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.images import EncodeOptions, ResizeOptions, encode

from ._common import client_headers, resolve_api_key, stream_ws_url
from .log import logger

DEFAULT_MODEL = "usyd-community/vitpose-plus-large"
DEFAULT_METHOD = "pose"

EventTypes = Literal["result", "frame_dropped", "error", "session_stats"]

FrameInput = rtc.VideoFrame | bytes

# The gateway bills the seconds a session is open, so a session with no new frame for this
# long (a paused feed, a muted camera) is closed and reopened on the next frame.
_IDLE_CLOSE_S = 5.0
# After max_retry back-to-back failures, wait this long before trying again.
_COOLDOWN_S = 30.0
# Open the next session this long before the current one expires.
_RENEW_MARGIN_S = 15.0
_MIN_SESSION_S = 5.0
_CLOSE_TIMEOUT_S = 5.0
# Fallbacks when session.created leaves a field out.
_DEFAULT_MAX_FPS = 5.0
_DEFAULT_MAX_FRAME_BYTES = 2 * 1024 * 1024
# Rejections that a reconnect cannot fix.
_FATAL_ERROR_TYPES = frozenset(
    {"authentication_error", "permission_error", "invalid_request_error", "not_found_error"}
)


@dataclass
class FrameResult:
    """The model's result for one frame."""

    frame: int
    """Sequence number assigned by :meth:`VisionStream.push_frame`."""
    timestamp: float
    """Wall-clock time (seconds since the epoch) when the frame was captured or pushed."""
    latency_ms: float | None
    """Server-side latency the gateway reported for this frame."""
    response: dict[str, Any]
    """The gateway's response, in the same shape as a chat completion."""

    @property
    def content(self) -> Any:
        """The message content, parsed from JSON when it is JSON. ``None`` if missing.

        For ViTPose ``pose`` this is ``{"object": "img.pose.kpts", "items": [...],
        "kpts_labels": [...]}``: one item per person with ``bbox_xywh``, ``label``,
        ``kpts_xy`` and ``kpts_score``, in image-relative coordinates that may fall outside
        0-1. HaMeR returns ``world.pose.kpts``, one item per hand. Shapes:
        https://docs.vlm.run/gateway/response-types
        """
        choices = self.response.get("choices") or []
        if not choices:
            return None
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return content


@dataclass
class FrameDropped:
    """A frame that was sent but never got a result."""

    frame: int
    timestamp: float
    reason: Literal["rate", "busy", "lost", "too_large"]
    """``rate`` and ``busy``: the gateway dropped it (sent faster than ``max_fps``, or a
    read was still running). ``lost``: the session ended first. ``too_large``: the encoded
    image exceeded the gateway's frame size limit, so it was never sent."""


@dataclass
class VisionStreamError:
    """An error reported by the gateway or the connection."""

    error: Exception
    error_type: str
    frame: int | None
    recoverable: bool
    """False when the stream has stopped and will not reconnect."""


@dataclass
class _Pending:
    frame: int
    timestamp: float
    data: FrameInput


class VisionStream(rtc.EventEmitter[EventTypes]):
    """Live video results from a VLM Run vision model, over the gateway's realtime socket.

    One WebSocket session (``wss://gateway.vlm.run/v1/ws``) per camera: it opens on the
    camera's first frame and closes when the track goes away or no frame has arrived for
    5 s, since the gateway bills the seconds a session is open. Connection failures are
    retried; only a refused model or key stops the stream. Only the newest frame is kept, it is
    JPEG-encoded when the previous read has finished, and sends are paced to the
    ``max_fps`` the gateway grants, so a slow result never queues stale frames.

    Only vision models that return a structured result per frame are served, such as
    ``usyd-community/vitpose-plus-large`` (method ``pose``) and ``geopavlakos/hamer``
    (method ``pose``). Use chat completions (:class:`LLM`) for a language answer.

    Events:
        result: :class:`FrameResult`
        frame_dropped: :class:`FrameDropped`
        error: :class:`VisionStreamError`
        session_stats: ``dict`` with the gateway's ``session.stats`` message
            (reads, dropped, errors, cost, ``p50_ms``, ``p95_ms``, fps)
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        method: str = DEFAULT_METHOD,
        width: int = 640,
        height: int = 640,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a VisionStream.

        Args:
            model: A vision model the gateway serves on its realtime socket.
            method: The model's method, e.g. ``pose``.
            width: Maximum width of the encoded frame, aspect ratio preserved.
            height: Maximum height of the encoded frame, aspect ratio preserved.
            api_key: VLM Run API key. Defaults to ``VLMRUN_API_KEY``. The socket requires one.
            base_url: Gateway URL. Defaults to ``VLMRUN_GATEWAY_BASE_URL`` or
                ``https://gateway.vlm.run``.
            http_session: Session to open the WebSocket with. Defaults to the job's.
            conn_options: Connection timeout, retry count and retry interval.
        """
        super().__init__()
        self._model = model
        self._method = method
        self._resize = ResizeOptions(width=width, height=height, strategy="scale_aspect_fit")
        self._api_key = resolve_api_key(api_key)
        self._url = stream_ws_url(base_url)
        self._conn_options = conn_options

        self._http_session = http_session
        self._owns_http_session = False

        self._seq = itertools.count()
        self._pending: _Pending | None = None
        self._frame_ready = asyncio.Event()
        self._closed = False
        # Set when the stream stops on its own after a fatal error; aclose() still runs.
        self._stopped = False
        self._main_task: asyncio.Task[None] | None = None
        self._stats: dict[str, Any] = {}
        # Results in the current session; a session that answered was healthy, so its
        # failure doesn't count toward back-to-back retries.
        self._session_results = 0

        self._room: rtc.Room | None = None
        self._participant_identity: str | None = None
        self._track_sid: str | None = None
        self._following_camera = False
        self._video_stream: rtc.VideoStream | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def model(self) -> str:
        return self._model

    @property
    def stats(self) -> dict[str, Any]:
        """The last ``session.stats`` message the gateway sent, or an empty dict."""
        return dict(self._stats)

    async def start(
        self,
        room: rtc.Room,
        participant: rtc.RemoteParticipant | str | None = None,
    ) -> None:
        """Read video from a room.

        The session opens when the participant's video track is subscribed and closes when
        it is unsubscribed.

        Args:
            room: A connected room.
            participant: The participant, or its identity, whose camera or screen share
                to read. Defaults to the first remote participant that publishes video.
        """
        if self._closed:
            raise RuntimeError("VisionStream is closed")
        if self._room is not None:
            raise RuntimeError("VisionStream is already attached to a room")

        self._room = room
        if isinstance(participant, rtc.RemoteParticipant):
            self._participant_identity = participant.identity
        else:
            self._participant_identity = participant

        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)

        self._follow_existing()

    def push_frame(self, frame: FrameInput, *, timestamp: float | None = None) -> int:
        """Offer one frame. It replaces any frame still waiting to be sent.

        Use this to feed frames that don't come from a room track, such as a robot's
        camera. Pushing faster than the gateway's ``max_fps`` is fine: only the newest
        frame is sent.

        Args:
            frame: A video frame, or already-encoded JPEG, PNG, WebP or GIF bytes.
            timestamp: Capture time in seconds since the epoch. Defaults to now.

        Returns:
            The frame's sequence number, matching :attr:`FrameResult.frame`.
        """
        if self._closed:
            raise RuntimeError("VisionStream is closed")
        if self._stopped:
            raise RuntimeError("VisionStream stopped after a fatal error")

        index = next(self._seq)
        self._pending = _Pending(
            frame=index, timestamp=time.time() if timestamp is None else timestamp, data=frame
        )
        self._frame_ready.set()
        if self._main_task is None or self._main_task.done():
            self._main_task = asyncio.create_task(self._run(), name="vlmrun.VisionStream")
        return index

    async def aclose(self) -> None:
        """Stop reading, close the session after the read in flight finishes, and release
        the room."""
        if self._closed:
            return
        self._closed = True
        self._frame_ready.set()

        if self._room is not None:
            self._room.off("track_subscribed", self._on_track_subscribed)
            self._room.off("track_unsubscribed", self._on_track_unsubscribed)
        await self._stop_reader()

        if self._main_task is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._main_task), self._conn_options.timeout + _CLOSE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                await utils.aio.cancel_and_wait(self._main_task)

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._owns_http_session and self._http_session is not None:
            await self._http_session.close()

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO or track.sid == self._track_sid:
            return
        if (
            self._participant_identity is not None
            and participant.identity != self._participant_identity
        ):
            return
        is_camera = publication.source == rtc.TrackSource.SOURCE_CAMERA
        # Keep the track being read unless this one is a camera and that one isn't (a
        # screen share must not take over from the camera).
        if self._track_sid is not None and (self._following_camera or not is_camera):
            return
        self._follow(track, is_camera)

    def _follow(self, track: rtc.Track, is_camera: bool) -> None:
        self._release_reader()
        self._following_camera = is_camera
        self._track_sid = track.sid
        self._video_stream = rtc.VideoStream(track)
        self._reader_task = asyncio.create_task(
            self._read_track(self._video_stream), name="vlmrun.VisionStream.read_track"
        )

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.sid != self._track_sid:
            return
        self._track_sid = None
        self._following_camera = False
        self._release_reader()
        # Drop the waiting frame so the session closes (the gateway bills open seconds),
        # then pick up another video track still in the room, cameras first.
        self._pending = None
        self._frame_ready.set()
        self._follow_existing(exclude=track.sid)

    def _follow_existing(self, exclude: str | None = None) -> None:
        if self._room is None:
            return
        candidates = [
            (publication, remote)
            for remote in self._room.remote_participants.values()
            for publication in remote.track_publications.values()
            if publication.track is not None and publication.track.sid != exclude
        ]
        candidates.sort(key=lambda c: c[0].source != rtc.TrackSource.SOURCE_CAMERA)
        for publication, remote in candidates:
            assert publication.track is not None
            self._on_track_subscribed(publication.track, publication, remote)

    def _release_reader(self) -> None:
        # A separate task owns closing the stream: a reader cancelled before it ever ran
        # skips its own cleanup, which would leak the VideoStream and its frame buffer.
        task, stream = self._reader_task, self._video_stream
        self._reader_task, self._video_stream = None, None
        if task is not None or stream is not None:
            self._track_task(asyncio.create_task(_close_reader(task, stream)))

    async def _stop_reader(self) -> None:
        task, stream = self._reader_task, self._video_stream
        self._reader_task, self._video_stream = None, None
        await _close_reader(task, stream)

    @utils.log_exceptions(logger=logger)
    async def _read_track(self, stream: rtc.VideoStream) -> None:
        async for event in stream:
            # The reader is cancelled from a separate task, so it can still see a frame
            # after its track was released; that frame must not reopen a session.
            if self._closed or self._stopped or stream is not self._video_stream:
                return
            self.push_frame(event.frame)

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._owns_http_session = True
        return self._http_session

    def _encode(self, data: FrameInput, limit: int) -> bytes:
        if isinstance(data, rtc.VideoFrame):
            raw = encode(data, EncodeOptions(format="JPEG", resize_options=self._resize))
        else:
            raw = data
        if len(raw) > limit:
            raise ValueError(f"encoded frame exceeds the gateway's {limit} byte limit")
        return raw

    def _camera_present(self) -> bool:
        # Attached to a room: keep a session only while a camera track is followed.
        return self._room is None or self._track_sid is not None

    async def _run(self) -> None:
        try:
            await self._run_loop()
        except Exception as e:
            logger.exception("vlmrun vision stream failed")
            error = e if isinstance(e, APIError) else APIError(str(e), retryable=False)
            self._emit_error(error, frame=None, recoverable=False)
        finally:
            if not self._closed:
                self._stopped = True

    async def _run_loop(self) -> None:
        failures = 0
        while not self._closed:
            if not self._camera_present():
                # Attached to a room with no camera followed: nothing to read, so no
                # session (it would be billed). A frame left over from the old camera goes.
                self._pending = None
            if self._pending is None:
                self._frame_ready.clear()
                await self._frame_ready.wait()
                continue

            try:
                ws, created = await self._connect()
            except APIError as e:
                failures += 1
                if not e.retryable:
                    self._emit_error(e, frame=None, recoverable=False)
                    break
                self._emit_error(e, frame=None, recoverable=True)
                await self._back_off(failures, _retry_after(e))
                continue

            self._session_results = 0
            try:
                answered = await self._run_session(ws, created)
                failures = 0
            except APIError as e:
                failures = 1 if self._session_results else failures + 1
                if not e.retryable:
                    self._emit_error(e, frame=None, recoverable=False)
                    break
                self._emit_error(e, frame=None, recoverable=True)
                await self._back_off(failures, None)
            else:
                logger.debug("vlmrun vision session ended", extra={"answered": answered})
            finally:
                await ws.close()

    async def _back_off(self, failures: int, retry_after: float | None) -> None:
        # Retryable failures never stop the stream: an outage must not end vision for the
        # rest of the room. Past max_retry, the waiting frame is dropped and the stream
        # cools down so a long outage doesn't become a reconnect storm.
        if failures > self._conn_options.max_retry:
            self._pending = None
            await self._sleep_unless_closed(max(retry_after or 0.0, _COOLDOWN_S))
        else:
            await self._sleep_unless_closed(
                retry_after or self._conn_options._interval_for_retry(failures - 1)
            )

    async def _sleep_unless_closed(self, seconds: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._wait_closed(), seconds)

    async def _wait_closed(self) -> None:
        while not self._closed:
            self._frame_ready.clear()
            await self._frame_ready.wait()

    async def _connect(self) -> tuple[aiohttp.ClientWebSocketResponse, dict[str, Any]]:
        session = self._ensure_http_session()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    self._url, headers=client_headers(self._api_key), autoping=False
                ),
                self._conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                "VLM Run gateway refused the WebSocket"
                + (" (set VLMRUN_API_KEY)" if e.status == 401 and not self._api_key else ""),
                status_code=e.status,
                retryable=e.status not in (400, 401, 403, 404),
            ) from None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to the VLM Run gateway") from e

        create = {
            "type": "session.create",
            "model": self._model,
            "method": self._method,
            "max_inflight": 1,
        }
        try:
            await ws.send_str(json.dumps(create))
            msg = await asyncio.wait_for(ws.receive(), self._conn_options.timeout)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            await ws.close()
            raise APIConnectionError("VLM Run gateway did not acknowledge the session") from e

        if msg.type != aiohttp.WSMsgType.TEXT:
            await ws.close()
            raise _closed_error(ws)
        created = _parse(msg.data)
        if created is None:
            await ws.close()
            raise APIError("invalid session reply from the VLM Run gateway", retryable=True)
        if created.get("type") != "session.created":
            await ws.close()
            raise _event_error(created)
        return ws, created

    async def _run_session(
        self, ws: aiohttp.ClientWebSocketResponse, created: dict[str, Any]
    ) -> int:
        """Send frames until the camera goes away, the stream closes, or renewal is due.

        Returns the number of results received. Raises APIError when the session fails.
        """
        loop = asyncio.get_running_loop()
        max_fps = float(created.get("max_fps") or _DEFAULT_MAX_FPS)
        interval = 1.0 / max_fps if max_fps > 0 else 0.0
        max_bytes = int(created.get("max_frame_bytes") or _DEFAULT_MAX_FRAME_BYTES)
        expires_in = float(created.get("expires_in") or 300.0)
        renew_at = loop.time() + max(_MIN_SESSION_S, expires_in - _RENEW_MARGIN_S)

        receiver = _Receiver(ws, self._conn_options)
        recv_task = asyncio.create_task(receiver.run(self._handle_event))
        sent = 0
        next_send = 0.0
        idle_since = loop.time()
        try:
            while True:
                if self._closed or not self._camera_present() or loop.time() >= renew_at:
                    break
                if recv_task.done():
                    recv_task.result()
                    raise APIConnectionError("VLM Run gateway closed the session")

                if self._pending is None:
                    idle_close_at = idle_since + _IDLE_CLOSE_S
                    if loop.time() >= idle_close_at:
                        break
                    self._frame_ready.clear()
                    timeout = max(0.0, min(renew_at, idle_close_at) - loop.time())
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(_first(self._frame_ready.wait(), recv_task), timeout)
                    continue

                # Pace to max_fps; newer frames replace the pending one while we wait.
                delay = next_send - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                    continue

                pending, self._pending = self._pending, None
                if pending is None:
                    continue
                try:
                    raw = await asyncio.to_thread(self._encode, pending.data, max_bytes)
                except ValueError:
                    self.emit(
                        "frame_dropped",
                        FrameDropped(pending.frame, pending.timestamp, "too_large"),
                    )
                    continue

                if recv_task.done():
                    # The gateway closed the session while the frame was being encoded.
                    self._pending = self._pending or pending
                    continue
                sent += 1
                outcome = receiver.expect(sent, pending)
                next_send = loop.time() + interval
                try:
                    await ws.send_bytes(raw)
                except (aiohttp.ClientError, ConnectionError) as e:
                    self._pending = self._pending or pending
                    raise APIConnectionError("VLM Run gateway connection lost") from e
                # One read in flight: wait for this frame's result, drop or error. A frame
                # the gateway never answers is reported lost after the request timeout.
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(_first(outcome, recv_task), self._conn_options.timeout)
                if not outcome.done():
                    receiver.drop_outstanding(self.emit)
                elif outcome.result():
                    self._session_results += 1
                if receiver.retry_after:
                    next_send = max(next_send, loop.time() + receiver.retry_after)
                    receiver.retry_after = 0.0
                idle_since = loop.time()
        finally:
            receiver.drop_outstanding(self.emit)
            if not recv_task.done():
                await self._close_session(ws, receiver, recv_task)
            await utils.aio.cancel_and_wait(recv_task)
        return self._session_results

    async def _close_session(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        receiver: _Receiver,
        recv_task: asyncio.Task[None],
    ) -> None:
        receiver.closing = True
        with contextlib.suppress(aiohttp.ClientError, ConnectionError):
            await ws.send_str(json.dumps({"type": "session.close"}))
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(recv_task), _CLOSE_TIMEOUT_S)

    def _handle_event(self, event: dict[str, Any], pending: _Pending | None) -> bool:
        """Emit the event. Returns True when it answered the frame in flight with a result."""
        kind = event.get("type")
        if kind == "result":
            if pending is None:
                return False
            response = event.get("response")
            self.emit(
                "result",
                FrameResult(
                    frame=pending.frame,
                    timestamp=pending.timestamp,
                    latency_ms=event.get("latency_ms"),
                    response=response if isinstance(response, dict) else {},
                ),
            )
            return True
        if kind == "frame.dropped":
            if pending is not None:
                reason = event.get("reason")
                self.emit(
                    "frame_dropped",
                    FrameDropped(
                        pending.frame,
                        pending.timestamp,
                        reason if reason in ("rate", "busy") else "busy",
                    ),
                )
            return False
        if kind == "error":
            error = _event_error(event)
            if not error.retryable and pending is None:
                raise error
            self._emit_error(error, frame=pending.frame if pending else None, recoverable=True)
            return False
        if kind == "session.stats":
            self._stats = event
            self.emit("session_stats", dict(event))
        return False

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _emit_error(self, error: APIError, *, frame: int | None, recoverable: bool) -> None:
        error_type = (
            error.body.get("error_type", "api_error")
            if isinstance(error.body, dict)
            else type(error).__name__
        )
        logger.warning(
            "vlmrun vision stream error",
            extra={"error_type": error_type, "recoverable": recoverable},
        )
        self.emit(
            "error",
            VisionStreamError(
                error=error, error_type=str(error_type), frame=frame, recoverable=recoverable
            ),
        )


class _Receiver:
    """Reads gateway events and resolves the frame in flight by the gateway's frame number."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, conn_options: APIConnectOptions):
        self._ws = ws
        self._conn_options = conn_options
        self._expected: int | None = None
        self._pending: _Pending | None = None
        self._outcome: asyncio.Future[bool] | None = None
        self.closing = False
        self.retry_after = 0.0

    def expect(self, gateway_frame: int, pending: _Pending) -> asyncio.Future[bool]:
        # The gateway numbers binary frames 1, 2, ... per session, in send order.
        self._expected = gateway_frame
        self._pending = pending
        self._outcome = asyncio.get_running_loop().create_future()
        return self._outcome

    def drop_outstanding(self, emit: Callable[..., Any]) -> None:
        if self._pending is not None and self._outcome is not None and not self._outcome.done():
            emit(
                "frame_dropped", FrameDropped(self._pending.frame, self._pending.timestamp, "lost")
            )
            self._outcome.set_result(False)
        self._pending = None

    async def run(self, handle: Callable[[dict[str, Any], _Pending | None], bool]) -> None:
        monitor = utils.aio.WebSocketMonitor(
            self._ws, liveness_timeout=self._conn_options.liveness_timeout
        )
        while True:
            msg = await monitor.receive()

            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                if self.closing:
                    return
                raise _closed_error(self._ws)
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(
                    "VLM Run gateway connection lost"
                ) from self._ws.exception()
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            event = _parse(msg.data)
            if event is None:
                logger.warning("invalid message from the VLM Run gateway")
                continue

            frame = event.get("frame")
            ours = isinstance(frame, int) and frame == self._expected
            pending = self._pending if ours else None
            answered = handle(event, pending)
            if event.get("type") == "error" and (retry_after := event.get("retry_after")):
                self.retry_after = float(retry_after)
            if ours and self._outcome is not None and not self._outcome.done():
                self._outcome.set_result(answered)
                self._pending = None
            if event.get("type") == "session.stats" and self.closing:
                return


async def _first(awaitable: Any, task: asyncio.Task[None]) -> None:
    """Wait for ``awaitable`` or until ``task`` finishes, whichever comes first."""
    waiter = asyncio.ensure_future(awaitable)
    # Only cancel a task created here. A future passed in (the frame's outcome) belongs
    # to the caller and must stay pending when the wait itself times out.
    owned = waiter is not awaitable
    try:
        await asyncio.wait([waiter, task], return_when=asyncio.FIRST_COMPLETED)
    finally:
        if owned and not waiter.done():
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter


async def _close_reader(task: asyncio.Task[None] | None, stream: rtc.VideoStream | None) -> None:
    if task is not None:
        await utils.aio.cancel_and_wait(task)
    if stream is not None:
        await stream.aclose()


def _parse(data: Any) -> dict[str, Any] | None:
    try:
        event = json.loads(data)
    except (TypeError, json.JSONDecodeError):
        return None
    return event if isinstance(event, dict) else None


def _retry_after(error: APIError) -> float | None:
    body = error.body if isinstance(error.body, dict) else {}
    value = body.get("retry_after")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _event_error(event: dict[str, Any]) -> APIStatusError:
    error_type = str(event.get("error_type") or "api_error")
    return APIStatusError(
        f"VLM Run gateway error ({error_type}): {event.get('message', '')}",
        status_code=-1,
        body=event,
        retryable=error_type not in _FATAL_ERROR_TYPES,
    )


def _closed_error(ws: aiohttp.ClientWebSocketResponse) -> APIStatusError:
    code = ws.close_code or -1
    # 1013 means the gateway is at capacity (too many sessions); worth retrying.
    return APIStatusError(
        f"VLM Run gateway closed the WebSocket (code {code})",
        status_code=code,
        retryable=code not in (1008,),
    )
