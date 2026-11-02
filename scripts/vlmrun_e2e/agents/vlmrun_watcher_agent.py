"""A voice agent that watches the user's camera and speaks up when something changes.

VLM Run reads every sampled frame over one WebSocket (``vlmrun.VisionStream``) and
answers typed questions about it. When the probability that a person is in frame
crosses a threshold, the agent says so.

Run with ``VLMRUN_API_KEY`` set, then join the room with a camera on:

    python vlmrun_watcher_agent.py dev
"""

import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, llm
from livekit.plugins import vlmrun

logger = logging.getLogger("vlmrun-watcher")

load_dotenv()

# Enter above ENTER, leave below LEAVE, so a flickering score doesn't repeat itself.
ENTER = 0.8
LEAVE = 0.3


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

    watcher = vlmrun.VisionStream(
        questions={
            "person": llm.Probability("A person is visible in the frame."),
            "activity": llm.Choice(
                "What is the person doing?",
                options={
                    "waving": "Waving at the camera",
                    "working": "Working at a desk or computer",
                    "other": "Anything else, or no person",
                },
            ),
        },
        state="A webcam pointed at a desk.",
        fps=2,
        publish_topic="vlmrun.decisions",
    )

    present = False

    @watcher.on("decision")
    def _on_decision(decision: vlmrun.FrameDecision) -> None:
        nonlocal present
        person = decision.values.get("person")
        if not isinstance(person, float):
            return
        if not present and person >= ENTER:
            present = True
            activity = decision.values.get("activity", "other")
            session.generate_reply(
                instructions=f"Someone just came into view (they look to be {activity}). "
                "Greet them briefly."
            )
        elif present and person <= LEAVE:
            present = False
            session.generate_reply(instructions="The person left the frame. Say so briefly.")

    @watcher.on("error")
    def _on_error(error: vlmrun.VisionStreamError) -> None:
        logger.warning("vision stream error", extra={"error_type": error.error_type})

    ctx.add_shutdown_callback(watcher.aclose)

    await session.start(agent=WatcherAgent(), room=ctx.room)
    await watcher.start(ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
