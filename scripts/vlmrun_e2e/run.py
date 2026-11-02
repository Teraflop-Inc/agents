"""End-to-end check of the VLM Run plugin in a real LiveKit Cloud room.

Not part of the PR: it lives on the vlmrun-e2e branch and runs from a workflow_dispatch job.

  1. plugin   A fake camera publishes a photo with people, then a blank frame. VisionStream
              (ViTPose over wss://gateway.vlm.run/v1/ws) must see people, then nobody, and
              close its session when the camera leaves.
  2. watcher  The camera shows a photo with people. The watcher agent, driven by pose
              results, must speak a greeting.
  3. camera   The camera shows the red circle. A chat message asks its color; the camera
              example (vlmrun.LLM) must answer "red".

Prints PASS/FAIL per check and exits non-zero on any failure.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from livekit import api, rtc
from livekit.agents import utils
from livekit.plugins import vlmrun

URL = os.environ["LIVEKIT_URL"]
KEY = os.environ["LIVEKIT_API_KEY"]
SECRET = os.environ["LIVEKIT_API_SECRET"]
REPO = Path(__file__).resolve().parents[2]
# The two agents live here, not in examples/, because the PR ships no examples.
EXAMPLES = Path(__file__).resolve().parent / "agents"
PEOPLE_PHOTO = "https://ultralytics.com/images/bus.jpg"
W, H = 640, 480

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def token(identity: str, room: str) -> str:
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


def red_stop() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((170, 40, 470, 340), fill="red")
    d.text((200, 360), "STOP", fill="black", font=ImageFont.load_default(size=96))
    return img


def blank() -> Image.Image:
    return Image.new("RGB", (W, H), "white")


def people() -> Image.Image:
    req = urllib.request.Request(PEOPLE_PHOTO, headers={"User-Agent": "vlmrun-e2e"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB").resize((W, H))


class Camera:
    """A participant that publishes whatever image is current, at 5 fps."""

    def __init__(self, image: Image.Image) -> None:
        self.room = rtc.Room()
        self.source = rtc.VideoSource(W, H)
        self.image = image
        self._task: asyncio.Task[None] | None = None

    async def start(self, room_name: str) -> None:
        await self.room.connect(URL, token("camera", room_name))
        track = rtc.LocalVideoTrack.create_video_track("camera", self.source)
        opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await self.room.local_participant.publish_track(track, opts)
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            rgba = self.image.convert("RGBA")
            frame = rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, rgba.tobytes())
            self.source.capture_frame(frame)
            await asyncio.sleep(0.2)

    async def stop(self) -> None:
        if self._task:
            await utils.aio.cancel_and_wait(self._task)
        await self.room.disconnect()


async def plugin_in_room() -> None:
    room_name = f"vlmrun-e2e-plugin-{uuid.uuid4().hex[:8]}"
    camera = Camera(people())
    await camera.start(room_name)
    observer = rtc.Room()
    await observer.connect(URL, token("observer", room_name))

    phase = "warmup"
    seen: list[tuple[str, int]] = []
    errors: list[str] = []
    stats: list[dict] = []
    stream = vlmrun.VisionStream(model="usyd-community/vitpose-plus-large", method="pose")
    stream.on(
        "result",
        lambda r: seen.append((phase, len((r.content or {}).get("items", [])))),
    )
    stream.on("error", lambda e: errors.append(f"{e.error_type}: {e.error}"))
    stream.on("session_stats", stats.append)
    await stream.start(observer)

    await asyncio.sleep(4)
    phase = "people"
    await asyncio.sleep(8)
    camera.image = blank()
    phase = "switching"
    await asyncio.sleep(2)
    phase = "blank"
    await asyncio.sleep(8)

    # The session must close when the camera leaves (the gateway bills open seconds).
    phase = "gone"
    await camera.stop()
    closed = await wait_until(lambda: bool(stats), 15)

    await stream.aclose()
    await observer.disconnect()

    with_people = [n for ph, n in seen if ph == "people"]
    empty = [n for ph, n in seen if ph == "blank"]
    check(
        "VisionStream: people photo returns people",
        len(with_people) >= 5 and all(n > 0 for n in with_people),
        f"n={len(with_people)} people per frame={sorted(set(with_people))}",
    )
    check(
        "VisionStream: blank frame returns nobody",
        len(empty) >= 5 and all(n == 0 for n in empty),
        f"n={len(empty)} people per frame={sorted(set(empty))}",
    )
    check(
        "VisionStream: session closes when the camera leaves",
        closed,
        json.dumps(stats[-1])[:200] if stats else "no session.stats",
    )
    check("VisionStream: no errors", not errors, "; ".join(errors[:3]))


def start_agent(example: str, room_name: str) -> subprocess.Popen[bytes]:
    log = open(f"/tmp/{example}.log", "wb")  # noqa: SIM115  # closed with the process
    return subprocess.Popen(
        [sys.executable, str(EXAMPLES / f"{example}.py"), "connect", "--room", room_name],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=EXAMPLES,
    )


def stop_agent(proc: subprocess.Popen[bytes], example: str) -> None:
    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log = Path(f"/tmp/{example}.log").read_text(errors="replace")
    print(f"--- {example} log: {len(log)} chars, full copy in /tmp/{example}.log ---")
    print(log[-3000:], flush=True)


async def collect_room(room: rtc.Room) -> tuple[list[str], list[dict]]:
    said: list[str] = []
    decisions: list[dict] = []

    async def _read(reader: rtc.TextStreamReader) -> None:
        text = await reader.read_all()
        if text.strip():
            said.append(text)

    tasks: set[asyncio.Task[None]] = set()

    def _on_text(reader: rtc.TextStreamReader, identity: str) -> None:
        task = asyncio.create_task(_read(reader))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def _on_data(packet: rtc.DataPacket) -> None:
        if packet.topic == "vlmrun.decisions":
            decisions.append(json.loads(packet.data))

    room.register_text_stream_handler("lk.transcription", _on_text)
    room.on("data_received", _on_data)
    return said, decisions


async def wait_until(pred, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await asyncio.sleep(0.5)
    return pred()


async def watcher_example() -> None:
    room_name = f"vlmrun-e2e-watcher-{uuid.uuid4().hex[:8]}"
    camera = Camera(people())
    said, _ = await collect_room(camera.room)
    await camera.start(room_name)
    proc = start_agent("vlmrun_watcher_agent", room_name)
    try:
        spoke = await wait_until(lambda: bool(said), 90)
        check(
            "watcher: greets when the pose model sees people",
            spoke,
            (said[:1] or [""])[0].strip()[:160],
        )
    finally:
        stop_agent(proc, "vlmrun_watcher_agent")
        await camera.stop()


async def camera_example() -> None:
    room_name = f"vlmrun-e2e-camera-{uuid.uuid4().hex[:8]}"
    # One participant publishes the camera, types and listens, as in a real app: the agent
    # links to a single participant, so a second one would race it for the agent's attention.
    camera = Camera(red_stop())
    said, _ = await collect_room(camera.room)
    await camera.start(room_name)
    proc = start_agent("vlmrun_camera_agent", room_name)
    try:
        greeted = await wait_until(lambda: bool(said), 90)
        check("camera: agent greets (vlmrun.LLM replied)", greeted, (said[:1] or [""])[0][:160])
        before = len(said)
        await camera.room.local_participant.send_text(
            "What color is the big round shape you see? Answer in one word.", topic="lk.chat"
        )
        answered = await wait_until(lambda: len(said) > before, 60)
        reply = " ".join(said[before:])
        check(
            "camera: answers about the frame (red)",
            answered and "red" in reply.lower(),
            reply[:160],
        )
    finally:
        stop_agent(proc, "vlmrun_camera_agent")
        await camera.stop()


async def main() -> int:
    async with utils.http_context.open():
        for part in (plugin_in_room, watcher_example, camera_example):
            print(f"=== {part.__name__} ===", flush=True)
            try:
                await part()
            except Exception as e:
                check(part.__name__, False, repr(e))
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
