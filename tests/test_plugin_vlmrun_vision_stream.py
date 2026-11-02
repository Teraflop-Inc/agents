from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents.types import APIConnectOptions
from livekit.plugins.vlmrun import vision_stream as vs
from livekit.plugins.vlmrun.vision_stream import FrameResult, VisionStream

pytestmark = pytest.mark.unit

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64

Reply = Callable[[web.WebSocketResponse, dict[str, Any]], Awaitable[None]]


def pose_response(people: int) -> dict[str, Any]:
    items = [{"bbox_xywh": [0.1, 0.1, 0.2, 0.5], "label": "person", "kpts_xy": []}] * people
    return {
        "object": "chat.completion",
        "model": "usyd-community/vitpose-plus-large",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"object": "img.pose.kpts", "items": items}),
                },
            }
        ],
    }


async def answer(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
    await ws.send_json(
        {
            "type": "result",
            "frame": event["frame"],
            "latency_ms": 45.0,
            "response": pose_response(1),
        }
    )


class FakeGateway:
    """Behaves like wss://gateway.vlm.run/v1/ws as probed live on 2026-11-02: binary frames
    are numbered 1, 2, ... per session, and result, frame.dropped and error name that number."""

    def __init__(
        self,
        reply: Reply = answer,
        *,
        created: dict[str, Any] | None = None,
        refuse: list[dict[str, Any]] | None = None,
        close_first_after: float | None = None,
    ) -> None:
        self.reply = reply
        self.created = {"max_fps": 50.0, "max_inflight": 1, "expires_in": 600.0, **(created or {})}
        self.refuse = list(refuse or [])
        self.close_first_after = close_first_after
        self.headers: list[dict[str, str]] = []
        self.creates: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self.closes: list[str] = []
        self.sessions = 0
        self.server = TestServer(self._app())

    def _app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/ws", self._handle)
        return app

    @property
    def base_url(self) -> str:
        return str(self.server.make_url("")).rstrip("/")

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.headers.append(dict(request.headers))
        self.creates.append(await ws.receive_json())
        if self.refuse:
            await ws.send_json({"type": "error", "frame": None, **self.refuse.pop(0)})
            await ws.close()
            return ws
        self.sessions += 1
        session_id = f"s{self.sessions}"
        await ws.send_json({"type": "session.created", "session_id": session_id, **self.created})
        if self.close_first_after is not None and self.sessions == 1:
            await asyncio.sleep(self.close_first_after)
            await ws.close(code=1013)  # at capacity, as the live gateway does
            return ws
        received = 0
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                received += 1
                frame = {"frame": received, "data": msg.data, "session": session_id}
                self.frames.append(frame)
                await self.reply(ws, frame)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                if json.loads(msg.data).get("type") == "session.close":
                    self.closes.append(session_id)
                    await ws.send_json({"type": "session.stats", "reads": received})
                    await ws.close()
            else:
                break
        return ws


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    monkeypatch.delenv("VLMRUN_API_KEY", raising=False)
    monkeypatch.delenv("VLMRUN_GATEWAY_BASE_URL", raising=False)


@pytest.fixture
async def http():
    async with aiohttp.ClientSession() as session:
        yield session


async def started(gateway: FakeGateway) -> FakeGateway:
    await gateway.server.start_server()
    return gateway


def make_stream(gateway: FakeGateway, http: aiohttp.ClientSession, **kwargs: Any) -> VisionStream:
    kwargs.setdefault("api_key", "vlm-key")
    kwargs.setdefault(
        "conn_options", APIConnectOptions(max_retry=1, retry_interval=0.05, timeout=2.0)
    )
    return VisionStream(base_url=gateway.base_url, http_session=http, **kwargs)


def collect(stream: VisionStream) -> dict[str, list[Any]]:
    events: dict[str, list[Any]] = {
        "result": [],
        "frame_dropped": [],
        "error": [],
        "session_stats": [],
    }
    for name, bucket in events.items():
        stream.on(name, bucket.append)
    return events


async def wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def test_session_create_and_binary_frame(http):
    gateway = await started(FakeGateway())
    try:
        stream = make_stream(gateway, http, model="geopavlakos/hamer", method="pose")
        events = collect(stream)
        index = stream.push_frame(JPEG, timestamp=12.5)
        await wait_for(lambda: len(events["result"]) == 1)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert gateway.creates[0] == {
        "type": "session.create",
        "model": "geopavlakos/hamer",
        "method": "pose",
        "max_inflight": 1,
    }
    assert gateway.headers[0]["Authorization"] == "Bearer vlm-key"
    assert gateway.headers[0]["X-VLMRun-Client"].startswith("livekit-agents/")
    assert gateway.frames[0]["data"] == JPEG, "frames go as raw binary, not base64"
    result: FrameResult = events["result"][0]
    assert (result.frame, result.timestamp, result.latency_ms) == (index, 12.5, 45.0)
    assert result.response["object"] == "chat.completion"
    assert result.content["object"] == "img.pose.kpts"
    assert len(result.content["items"]) == 1
    assert gateway.closes == ["s1"], "the session is closed politely"
    assert events["session_stats"][0]["reads"] == 1


async def test_one_read_in_flight_and_newest_frame_wins(http):
    release = asyncio.Event()
    background: set[asyncio.Future[None]] = set()

    async def slow_first(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        if event["frame"] == 1:

            async def later() -> None:
                await release.wait()
                await answer(ws, event)

            background.add(asyncio.ensure_future(later()))
        else:
            await answer(ws, event)

    gateway = await started(FakeGateway(slow_first))
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        first = stream.push_frame(JPEG)
        await wait_for(lambda: len(gateway.frames) == 1)
        stream.push_frame(JPEG + b"a")
        stream.push_frame(JPEG + b"b")
        newest = stream.push_frame(JPEG + b"c")
        await asyncio.sleep(0.2)
        assert len(gateway.frames) == 1, "no second read while one is in flight"
        release.set()
        await wait_for(lambda: len(events["result"]) == 2)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert [r.frame for r in events["result"]] == [first, newest]
    assert gateway.frames[1]["data"] == JPEG + b"c", "only the newest waiting frame is sent"


async def test_sends_are_paced_to_max_fps(http):
    gateway = await started(FakeGateway(created={"max_fps": 5.0}))
    sent_at: list[float] = []
    loop = asyncio.get_running_loop()

    async def timed(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        sent_at.append(loop.time())
        await answer(ws, event)

    gateway.reply = timed
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        for i in range(3):
            stream.push_frame(JPEG)
            await wait_for(lambda n=i: len(sent_at) > n)
        await stream.aclose()
    finally:
        await gateway.server.close()

    gaps = [b - a for a, b in zip(sent_at, sent_at[1:], strict=False)]
    assert all(gap >= 0.18 for gap in gaps), f"5 fps means 0.2 s apart: {gaps}"
    assert len(events["result"]) == 3


async def test_server_drop_is_reported_and_streaming_continues(http):
    calls = 0

    async def drop_first(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            await ws.send_json(
                {"type": "frame.dropped", "frame": event["frame"], "inflight": 1, "reason": "rate"}
            )
        else:
            await answer(ws, event)

    gateway = await started(FakeGateway(drop_first))
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        dropped = stream.push_frame(JPEG)
        await wait_for(lambda: len(events["frame_dropped"]) == 1)
        kept = stream.push_frame(JPEG)
        await wait_for(lambda: len(events["result"]) == 1)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert [(d.frame, d.reason) for d in events["frame_dropped"]] == [(dropped, "rate")]
    assert events["result"][0].frame == kept
    assert gateway.sessions == 1


async def test_bad_frame_error_is_recoverable(http):
    async def reject_garbage(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        if event["data"] == b"garbage":
            await ws.send_json(
                {
                    "type": "error",
                    "error_type": "invalid_request_error",
                    "message": "a frame is raw JPEG, PNG, WebP or GIF bytes",
                    "frame": event["frame"],
                    "retry_after": None,
                }
            )
        else:
            await answer(ws, event)

    gateway = await started(FakeGateway(reject_garbage))
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        bad = stream.push_frame(b"garbage")
        await wait_for(lambda: len(events["error"]) == 1)
        good = stream.push_frame(JPEG)
        await wait_for(lambda: len(events["result"]) == 1)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert [(e.frame, e.recoverable) for e in events["error"]] == [(bad, True)]
    assert events["result"][0].frame == good
    assert gateway.sessions == 1


async def test_generative_model_is_refused_without_retry(http):
    refusal = {
        "error_type": "invalid_request_error",
        "message": "Model 'qwen/qwen3.8-27b' is generative; a socket streams frame models.",
        "retry_after": None,
    }
    gateway = await started(FakeGateway(refuse=[refusal]))
    try:
        stream = make_stream(gateway, http, model="qwen/qwen3.8-27b", method="chat")
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: any(not e.recoverable for e in events["error"]))
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert len(gateway.creates) == 1, "no retry after a refusal"
    assert "generative" in str(events["error"][0].error)
    with pytest.raises(RuntimeError):
        stream.push_frame(JPEG)


async def test_cold_model_waits_retry_after_then_connects(http):
    cold = {
        "error_type": "model_provisioning_error",
        "message": "geopavlakos/hamer is starting up. Retry in about 240 s.",
        "retry_after": 0.3,
    }
    gateway = await started(FakeGateway(refuse=[cold]))
    loop = asyncio.get_running_loop()
    try:
        stream = make_stream(gateway, http, model="geopavlakos/hamer")
        events = collect(stream)
        started_at = loop.time()
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["result"]) == 1)
        waited = loop.time() - started_at
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert [(e.error_type, e.recoverable) for e in events["error"]] == [
        ("model_provisioning_error", True)
    ]
    assert waited >= 0.28, f"waited for retry_after: {waited:.2f}s"
    assert len(gateway.creates) == 2


async def test_unanswered_frame_is_lost_after_timeout(http):
    async def ignore_first(ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        if event["frame"] != 1:
            await answer(ws, event)

    gateway = await started(FakeGateway(ignore_first))
    try:
        stream = make_stream(
            gateway,
            http,
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0.05, timeout=0.3),
        )
        events = collect(stream)
        lost = stream.push_frame(JPEG)
        await wait_for(lambda: len(gateway.frames) == 1)
        later = stream.push_frame(JPEG)
        await wait_for(lambda: len(events["result"]) == 1)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert [(d.frame, d.reason) for d in events["frame_dropped"]] == [(lost, "lost")]
    assert events["result"][0].frame == later


async def test_session_renews_before_expiry(http, monkeypatch):
    monkeypatch.setattr(vs, "_MIN_SESSION_S", 0.2)
    monkeypatch.setattr(vs, "_RENEW_MARGIN_S", 0.0)
    gateway = await started(FakeGateway(created={"expires_in": 0.2}))
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["result"]) == 1)
        await asyncio.sleep(0.4)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["result"]) == 2)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert gateway.sessions == 2
    assert gateway.closes[0] == "s1", "the old session was closed politely"
    assert events["error"] == [], "renewal is not an error"


async def test_handshake_rejection_without_key_is_fatal(http):
    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=401)

    app = web.Application()
    app.router.add_get("/v1/ws", reject)
    server = TestServer(app)
    await server.start_server()
    try:
        stream = VisionStream(
            base_url=str(server.make_url("")).rstrip("/"),
            http_session=http,
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0.05, timeout=2.0),
        )
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: bool(events["error"]))
        await stream.aclose()
    finally:
        await server.close()

    assert not events["error"][0].recoverable
    assert "VLMRUN_API_KEY" in str(events["error"][0].error)


async def test_aclose_after_fatal_error_releases_owned_session():
    refusal = {"error_type": "authentication_error", "message": "bad key", "retry_after": None}
    gateway = await started(FakeGateway(refuse=[refusal]))
    try:
        stream = VisionStream(
            api_key="bad",
            base_url=gateway.base_url,
            conn_options=APIConnectOptions(max_retry=0, retry_interval=0.05, timeout=2.0),
        )
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: any(not e.recoverable for e in events["error"]))
        owned = stream._http_session
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert owned is not None and owned.closed, "aclose() still cleans up after the stream stopped"


class _FakeVideoStream:
    """Yields the same frame until closed, like a camera track."""

    def __init__(self, track: Any) -> None:
        self.sid = track.sid
        self.closed = False

    def __aiter__(self) -> _FakeVideoStream:
        return self

    async def __anext__(self) -> Any:
        if self.closed:
            raise StopAsyncIteration
        await asyncio.sleep(0.02)
        event = MagicMock()
        event.frame = JPEG
        return event

    async def aclose(self) -> None:
        self.closed = True


class _FakeRoom:
    def __init__(self) -> None:
        self.remote_participants: dict[str, Any] = {}
        self.handlers: dict[str, Any] = {}

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    def off(self, event: str, handler: Any) -> None:
        self.handlers.pop(event, None)


class _Pub:
    def __init__(self, track: Any, source: Any) -> None:
        self.track = track
        self.source = source


def _camera(sid: str) -> Any:
    return MagicMock(kind=rtc.TrackKind.KIND_VIDEO, sid=sid)


async def test_session_follows_the_camera_track(http, monkeypatch):
    """Opens when the camera arrives, closes when it leaves (the gateway bills open seconds)."""
    monkeypatch.setattr(vs.rtc, "VideoStream", _FakeVideoStream)
    gateway = await started(FakeGateway())
    room = _FakeRoom()
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        await stream.start(room)  # type: ignore[arg-type]
        camera = _camera("cam")
        participant = MagicMock(identity="user")
        room.handlers["track_subscribed"](camera, MagicMock(), participant)
        await wait_for(lambda: len(events["result"]) >= 2)
        room.handlers["track_unsubscribed"](camera, MagicMock(), participant)
        await wait_for(lambda: gateway.closes == ["s1"])
        sent = len(gateway.frames)
        await asyncio.sleep(0.3)
        assert len(gateway.frames) == sent, "no frames after the camera left"
        assert gateway.sessions == 1, "no new session after the camera left"
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert room.handlers == {}, "room handlers removed"


async def test_every_followed_track_stream_is_closed(http, monkeypatch):
    """Tracks switched within one tick: a reader that never ran still closes its stream."""
    closed: list[str] = []

    class Recording(_FakeVideoStream):
        async def aclose(self) -> None:
            closed.append(self.sid)
            await super().aclose()

    monkeypatch.setattr(vs.rtc, "VideoStream", Recording)
    gateway = await started(FakeGateway())
    try:
        stream = make_stream(gateway, http)
        user = MagicMock(identity="user")
        screen = _Pub(_camera("screen"), rtc.TrackSource.SOURCE_SCREENSHARE)
        camera = _Pub(_camera("cam"), rtc.TrackSource.SOURCE_CAMERA)
        stream._on_track_subscribed(screen.track, screen, user)  # type: ignore[arg-type]
        stream._on_track_subscribed(camera.track, camera, user)  # type: ignore[arg-type]
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert sorted(closed) == ["cam", "screen"]


async def test_screen_share_does_not_take_over_and_camera_is_resumed(http, monkeypatch):
    monkeypatch.setattr(vs.rtc, "VideoStream", _FakeVideoStream)
    gateway = await started(FakeGateway())
    room = _FakeRoom()
    camera, screen = _camera("cam"), _camera("screen")
    user = MagicMock(identity="user")
    user.track_publications = {}
    room.remote_participants = {"user": user}
    try:
        stream = make_stream(gateway, http)
        await stream.start(room, participant="user")  # type: ignore[arg-type]
        camera_pub = _Pub(camera, rtc.TrackSource.SOURCE_CAMERA)
        screen_pub = _Pub(screen, rtc.TrackSource.SOURCE_SCREENSHARE)
        user.track_publications = {"cam": camera_pub}
        room.handlers["track_subscribed"](camera, camera_pub, user)
        assert stream._track_sid == "cam"
        user.track_publications = {"cam": camera_pub, "screen": screen_pub}
        room.handlers["track_subscribed"](screen, screen_pub, user)
        assert stream._track_sid == "cam", "a screen share must not replace the camera"

        # Camera leaves while the screen share stays: fall back to the screen share.
        room.handlers["track_unsubscribed"](camera, camera_pub, user)
        assert stream._track_sid == "screen"
        # Camera comes back: it takes over from the screen share.
        room.handlers["track_subscribed"](camera, camera_pub, user)
        assert stream._track_sid == "cam"
        await stream.aclose()
    finally:
        await gateway.server.close()


async def test_no_session_without_a_camera(http):
    """Attached to a room with no camera followed, a frame must not open (billed) sessions."""
    gateway = await started(FakeGateway())
    room = _FakeRoom()
    try:
        stream = make_stream(gateway, http)
        await stream.start(room)  # type: ignore[arg-type]
        stream.push_frame(JPEG)
        await asyncio.sleep(0.5)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert gateway.creates == [], f"opened {len(gateway.creates)} sessions with no camera"


async def test_idle_session_closes_and_reopens_on_next_frame(http, monkeypatch):
    monkeypatch.setattr(vs, "_IDLE_CLOSE_S", 0.3)
    gateway = await started(FakeGateway())
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["result"]) == 1)
        await wait_for(lambda: gateway.closes == ["s1"], timeout=2.0)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["result"]) == 2)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert gateway.sessions == 2


async def test_outage_past_max_retry_cools_down_without_stopping(http, monkeypatch):
    """A gateway outage must not end vision for the rest of the room."""
    monkeypatch.setattr(vs, "_COOLDOWN_S", 0.3)
    overloaded = {"error_type": "overloaded_error", "message": "busy", "retry_after": None}
    gateway = await started(FakeGateway(refuse=[overloaded] * 4))
    try:
        stream = make_stream(gateway, http)  # max_retry=1
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(gateway.creates) >= 2)
        await asyncio.sleep(0.1)
        attempts_before_cooldown = len(gateway.creates)
        for _ in range(40):  # a camera keeps offering frames
            stream.push_frame(JPEG)
            if events["result"]:
                break
            await asyncio.sleep(0.1)
        await wait_for(lambda: len(events["result"]) == 1, timeout=5.0)
        stopped = stream._stopped
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert attempts_before_cooldown == 2, "cooled down after max_retry instead of hammering"
    assert all(e.recoverable for e in events["error"]), "retryable errors never stop it"
    assert not stopped, "the stream is still running"


async def test_gateway_close_while_encoding_reconnects(http, monkeypatch):
    """The socket closing under a send (e.g. 1013 at capacity) must reconnect, not stop."""
    original = VisionStream._encode

    def slow_encode(self: VisionStream, data: Any, limit: int) -> bytes:
        import time

        time.sleep(0.3)
        return original(self, data, limit)

    monkeypatch.setattr(VisionStream, "_encode", slow_encode)
    gateway = await started(FakeGateway(close_first_after=0.1))
    try:
        stream = make_stream(gateway, http)
        events = collect(stream)
        stream.push_frame(JPEG)
        await wait_for(lambda: len(events["result"]) == 1, timeout=5.0)
        await stream.aclose()
    finally:
        await gateway.server.close()

    assert all(e.recoverable for e in events["error"]), events["error"]
    assert gateway.sessions == 2
