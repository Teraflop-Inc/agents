from __future__ import annotations

import asyncio
import io
import os

import aiohttp
import pytest

from livekit.plugins.vlmrun.vision_stream import FrameResult, VisionStream

pytestmark = [
    pytest.mark.plugin("vlmrun"),
    # The realtime socket refuses connections without a key (401).
    pytest.mark.skipif(not os.environ.get("VLMRUN_API_KEY"), reason="VLMRUN_API_KEY not set"),
]

PEOPLE_PHOTO = "https://ultralytics.com/images/bus.jpg"


async def _jpegs(http: aiohttp.ClientSession) -> tuple[bytes, bytes]:
    from PIL import Image

    # The host serves this photo as application/octet-stream; read it as bytes.
    async with http.get(PEOPLE_PHOTO) as resp:
        resp.raise_for_status()
        photo = Image.open(io.BytesIO(await resp.read())).convert("RGB").resize((640, 480))

    def encode(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, "JPEG")
        return buf.getvalue()

    return encode(photo), encode(Image.new("RGB", (640, 480), "white"))


async def test_pose_results_follow_frame_content():
    async with aiohttp.ClientSession() as http:
        people, blank = await _jpegs(http)
        stream = VisionStream(http_session=http)
        results: dict[int, FrameResult] = {}
        dropped: list[int] = []
        errors: list[str] = []
        stream.on("result", lambda r: results.__setitem__(r.frame, r))
        stream.on("frame_dropped", lambda d: dropped.append(d.frame))
        stream.on("error", lambda e: errors.append(f"{e.error_type}: {e.error}"))

        # Opening the session takes a moment and only the newest frame is kept, so warm up
        # before the frames under test.
        stream.push_frame(blank)

        async def settled(count: int) -> None:
            while len(results) + len(dropped) < count:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(settled(1), 30)
        pushed = []
        for i in range(6):
            pushed.append((stream.push_frame(people if i % 2 == 0 else blank), i % 2 == 0))
            await asyncio.sleep(0.5)
        await asyncio.wait_for(settled(1 + len(pushed)), 30)
        await stream.aclose()

    assert not errors, errors
    answered = [(frame, has_people) for frame, has_people in pushed if frame in results]
    assert len(answered) >= len(pushed) - 1, f"too many frames dropped: {dropped}"
    for frame, has_people in answered:
        items = results[frame].content["items"]
        if has_people:
            assert items and items[0]["label"] == "person", f"frame {frame}: {items[:1]}"
        else:
            assert items == [], f"frame {frame} is blank: {items[:1]}"
