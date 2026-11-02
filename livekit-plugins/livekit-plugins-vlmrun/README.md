# VLM Run plugin for LiveKit Agents

Vision for LiveKit agents through the [VLM Run gateway](https://docs.vlm.run/gateway/introduction):
an LLM that reads camera frames, and live video results from vision models such as pose
estimation.

## Installation

```bash
pip install livekit-plugins-vlmrun
```

## Pre-requisites

Set `VLMRUN_API_KEY` (from [app.vlm.run](https://app.vlm.run)). Without a key, the chat route uses
the gateway's anonymous tier, at lower rate limits; some models require a paid account.
Live video (`VisionStream`) always needs a key. Set `VLMRUN_GATEWAY_BASE_URL` to use another
gateway.

## LLM

The gateway's OpenAI-compatible chat route. Images in the chat context, including video frames,
are sent to the model.

```python
from livekit.agents import AgentSession
from livekit.plugins import vlmrun

session = AgentSession(llm=vlmrun.LLM(model="qwen/qwen3.8-27b"), stt=..., tts=...)
```

Tool calling works on `qwen/qwen3.8-27b` and the provider models (`google/gemini-*`,
`moonshotai/kimi-k3`, `minimax/minimax-m3`); the provider models need a paid account. Sampling
settings are sent only when set, because the gateway rejects settings a model doesn't support.

## Live video

`VisionStream` sends a room's camera to a VLM Run vision model over the gateway's realtime socket
(`wss://gateway.vlm.run/v1/ws`) and emits the model's result for each frame it reads. The gateway
bills the seconds a session is open, so the session opens on the camera's first frame and closes
when the camera leaves or no frame has arrived for 5 seconds.

```python
from livekit.plugins import vlmrun

stream = vlmrun.VisionStream(model="usyd-community/vitpose-plus-large", method="pose")


@stream.on("result")
def on_result(result: vlmrun.FrameResult) -> None:
    people = (result.content or {}).get("items", [])
    if people:
        ...  # a person is in view


await stream.start(ctx.room)
```

Only the newest frame is kept and one read is in flight at a time, paced to the `max_fps` the
gateway grants, so a slow result never queues stale frames. `result.response` is in the same shape
as a chat completion; `result.content` is its parsed content. The socket serves vision models that
return a structured result per frame, such as ViTPose and HaMeR (`geopavlakos/hamer`, method
`pose`); use `vlmrun.LLM` for a language answer about a result. Use `push_frame()` to feed frames
that don't come from a room, such as a robot camera.
