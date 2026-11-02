"""Voice agent that answers questions about what the user's camera sees, using VLM Run.

The agent keeps the latest frame from the user's camera (or screen share) and attaches it to
each reply, spoken or typed, so the VLM Run model answers with the current view in front of it.

Set VLMRUN_API_KEY for production rate limits. Without it the gateway's anonymous tier is
used, at lower rate limits.

Run with:
    python vlmrun_camera_agent.py console   # local mic and speaker, no camera
    python vlmrun_camera_agent.py dev       # join a LiveKit room and publish a camera
"""

import asyncio
import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    FlushSentinel,
    JobContext,
    ModelSettings,
    cli,
    get_job_context,
    inference,
    llm,
)
from livekit.plugins import vlmrun

logger = logging.getLogger("vlmrun-camera-agent")

load_dotenv()


class CameraAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant that can see through the user's camera. "
            "Each user message may include the current camera frame. "
            "Answer questions about what you see in one or two short spoken sentences. "
            "Do not use markdown, emojis or special characters. "
            "If no image is attached, say you can't see anything yet.",
        )
        self._latest_frame: rtc.VideoFrame | None = None
        self._video_stream: rtc.VideoStream | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def on_enter(self) -> None:
        room = get_job_context().room

        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                    self._watch(publication.track)
                    break

        room.on("track_subscribed", self._on_track_subscribed)

        self.session.generate_reply(
            instructions="Greet the user and offer to describe what you see."
        )

    async def on_exit(self) -> None:
        get_job_context().room.off("track_subscribed", self._on_track_subscribed)
        if self._video_stream is not None:
            await self._video_stream.aclose()
            self._video_stream = None

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self._watch(track)

    def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
        # Attach the frame here rather than in on_user_turn_completed: typed chat messages
        # (generate_reply(user_input=...)) skip that hook. Only this request gets the image,
        # so old frames don't pile up in the conversation history.
        if self._latest_frame is not None:
            chat_ctx = chat_ctx.copy()
            for i in range(len(chat_ctx.items) - 1, -1, -1):
                item = chat_ctx.items[i]
                if item.type == "message" and item.role == "user":
                    # 768px keeps detail for small text while bounding image tokens per turn.
                    frame = llm.ImageContent(
                        image=self._latest_frame, inference_width=768, inference_height=768
                    )
                    chat_ctx.items[i] = item.model_copy(update={"content": [*item.content, frame]})
                    break
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    def _watch(self, track: rtc.Track) -> None:
        if self._video_stream is not None:
            old = self._video_stream
            task = asyncio.create_task(old.aclose())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        stream = rtc.VideoStream(track)
        self._video_stream = stream

        async def _read() -> None:
            async for event in stream:
                self._latest_frame = event.frame

        task = asyncio.create_task(_read())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session: AgentSession = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=vlmrun.LLM(model="qwen/qwen3.8-27b"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )

    await session.start(agent=CameraAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
