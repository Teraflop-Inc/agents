"""A voice agent that watches the user's camera and speaks up when someone comes into view.

VLM Run's pose model reads the camera over the gateway's realtime socket
(``vlmrun.VisionStream``). When people appear in its results, the agent greets them; when
they have been gone for a few frames, it says so.

Run with ``VLMRUN_API_KEY`` set, then join the room with a camera on:

    python vlmrun_watcher_agent.py dev
"""

import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import vlmrun

logger = logging.getLogger("vlmrun-watcher")

load_dotenv()

# Frames without anyone before the agent says the person left, so a missed frame doesn't.
GONE_AFTER = 5


class WatcherAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You watch a camera feed and talk to the user by voice. "
            "Keep replies to one short sentence. No markdown or emojis."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    watcher = vlmrun.VisionStream(model="usyd-community/vitpose-plus-large", method="pose")
    present = False
    empty_frames = 0

    @watcher.on("result")
    def _on_result(result: vlmrun.FrameResult) -> None:
        nonlocal present, empty_frames
        people = len((result.content or {}).get("items", []))
        if people:
            empty_frames = 0
            if not present:
                present = True
                session.generate_reply(
                    instructions=f"{people} person(s) just came into view. Greet them briefly."
                )
        elif present:
            empty_frames += 1
            if empty_frames >= GONE_AFTER:
                present = False
                session.generate_reply(instructions="Everyone left the frame. Say so briefly.")

    @watcher.on("error")
    def _on_error(error: vlmrun.VisionStreamError) -> None:
        logger.warning("vision stream error", extra={"error_type": error.error_type})

    ctx.add_shutdown_callback(watcher.aclose)

    await session.start(agent=WatcherAgent(), room=ctx.room)
    await watcher.start(ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
