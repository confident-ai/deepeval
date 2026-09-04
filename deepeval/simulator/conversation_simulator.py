from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Optional,
    List,
    Tuple,
    Type,
    Union,
    Callable,
)
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime
from rich.progress import Progress
from pydantic import BaseModel
import inspect
import asyncio
import logging
import time
import uuid
import warnings

from deepeval.utils import (
    get_or_create_event_loop,
    update_pbar,
    add_pbar,
)
from deepeval.metrics.utils import (
    initialize_model,
    trimAndLoadJson,
)
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.simulator.template import (
    SimulationTemplate,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import MULTIMODAL_SUPPORTED_MODELS
from deepeval.simulator.schema import (
    SimulatedInput,
)
from deepeval.simulator.controller.controller import (
    SimulationController,
    expected_outcome_controller,
)
from deepeval.simulator.simulation_graph import (
    SimulationNode,
    default_simulation_node,
)
from deepeval.simulator.simulation_graph.runner import (
    _SimulationGraphRunner,
    _GraphConversationState,
)
from deepeval.progress_context import conversation_simulator_progress_context
from deepeval.voice.recording import CallRecorder, RecordingConnector
from deepeval.dataset import ConversationalGolden

if TYPE_CHECKING:
    from deepeval.dataset import Persona
    from deepeval.models.base_model import DeepEvalBaseSTT, DeepEvalBaseTTS
    from deepeval.test_case import Audio
    from deepeval.voice.connectors.transports.base import BaseVoiceConnector
    from deepeval.voice.config import VoiceConfig
    from deepeval.voice.floor_control import FloorController
    from deepeval.voice.interruption import InterruptionPolicy

logger = logging.getLogger(__name__)

_MISSING = object()

# Length of the silent uplink used to hear the agent out without speaking.
_SILENCE_PROBE_SECONDS = 1.0
_GREETING_TIMEOUT_S = 15.0


@dataclass
class _VoiceSession:
    """One conversation's live session: its connector, caller, and floor state."""

    connector: "BaseVoiceConnector"
    persona: Optional["Persona"]
    policy: Optional["InterruptionPolicy"] = None
    floor: Optional["FloorController"] = None
    started_at: Optional[float] = None
    barges: int = 0
    recorder: Optional[CallRecorder] = None

    @property
    def is_duplex(self) -> bool:
        """A barge policy or a full-duplex line both take the duplex path."""
        return self.policy is not None or self.connector.supports_duplex


@dataclass
class _VoiceRun:
    """A simulator's voice mode: the speech models and the spend.

    Present only when the simulator was given a `VoiceConfig` — its absence is
    what text mode is. The speech models last as long as the simulator, the
    costs and the run label restart on each `simulate()`, and every
    conversation gets its own `_VoiceSession` with a connector of its own.
    """

    config: "VoiceConfig"
    tts_model: "DeepEvalBaseTTS"
    stt_model: "DeepEvalBaseSTT"

    # Per `simulate()` call.
    tts_cost: float = 0.0
    stt_cost: float = 0.0
    run_timestamp: Optional[str] = None
    num_goldens: int = 0

    def begin_run(self, num_goldens: int) -> None:
        self.tts_cost = 0.0
        self.stt_cost = 0.0
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.num_goldens = num_goldens

    def open_session(
        self,
        persona: Optional["Persona"],
        policy: Optional["InterruptionPolicy"],
        floor: Optional["FloorController"],
    ) -> _VoiceSession:
        from deepeval.voice.floor_control import FloorController

        connector = self.config.make_connector()
        recorder = CallRecorder() if self.config.record_call else None
        if recorder is not None:
            connector = RecordingConnector(connector, recorder)
        return _VoiceSession(
            connector=connector,
            persona=persona,
            policy=policy,
            floor=floor or FloorController(),
            recorder=recorder,
        )

    @property
    def run_label(self) -> str:
        return "simulation-{}".format(
            self.run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )


def _has_speech_energy(audio, connector) -> bool:
    """Whether the clip contains anything above the connector's silence floor.

    Silence must never reach STT: models hallucinate words on silent audio,
    and a phantom transcript makes a dead agent look alive to both the
    hold timeout and the metrics.
    """
    from deepeval.voice.connectors import audio_utils

    try:
        pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
    except Exception:
        return True
    pcm = audio_utils.downmix_to_mono(pcm, channels or 1)
    threshold = getattr(
        connector, "silence_threshold_rms", audio_utils.DEFAULT_SILENCE_RMS
    )
    frame_bytes = 2 * int(rate * audio_utils.DEFAULT_FRAME_MS / 1000)
    for offset in range(0, len(pcm), frame_bytes):
        if not audio_utils.is_silent(
            pcm[offset : offset + frame_bytes], threshold
        ):
            return True
    return False


def _populate_audio_duration(audio) -> None:
    if audio is None or audio.duration is not None:
        return
    try:
        from deepeval.voice.connectors import audio_utils

        pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(
            audio.get_bytes()
        )
    except (TypeError, ValueError):
        return
    if sample_rate:
        audio.duration = (len(pcm) / 2 / max(channels, 1)) / sample_rate


OnTurnCallback = Callable[[List[Turn], int], Union[None, Awaitable[None]]]


async def _dispatch_on_turn(
    on_turn: Optional[OnTurnCallback], turns: List[Turn], index: Optional[int]
) -> None:
    if on_turn is None:
        return
    try:
        result = on_turn(list(turns), index if index is not None else 0)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("on_turn hook failed")


def _dispatch_on_turn_sync(
    on_turn: Optional[OnTurnCallback], turns: List[Turn], index: Optional[int]
) -> None:
    if on_turn is None:
        return
    try:
        result = on_turn(list(turns), index if index is not None else 0)
        if inspect.isawaitable(result):
            asyncio.run(result)
    except Exception:
        logger.exception("on_turn hook failed")


async def _discard_task(task: Optional[asyncio.Task]) -> None:
    """Drop work started ahead of a decision that turned out not to need it."""
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Discarded simulated user generation failed")


class ConversationSimulator:
    def __init__(
        self,
        model_callback: Optional[Callable[[str], str]] = None,
        simulation_graph: Optional[SimulationNode] = None,
        stopping_controller: Callable = expected_outcome_controller,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        max_concurrent: int = 5,
        async_mode: bool = True,
        language: str = "English",
        voice_config: Optional["VoiceConfig"] = None,
        controller: Any = _MISSING,
    ):
        if controller is not _MISSING:
            if stopping_controller is not expected_outcome_controller:
                raise TypeError(
                    "Pass either `stopping_controller` or the deprecated "
                    "`controller`, not both."
                )
            warnings.warn(
                "`controller` is deprecated; use `stopping_controller` "
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            stopping_controller = controller

        if (model_callback is None) == (voice_config is None):
            raise TypeError(
                "Provide exactly one of `model_callback` (text agents) or "
                "`voice_config` (voice agents)."
            )

        self._voice: Optional[_VoiceRun] = None
        if voice_config is not None:
            if not async_mode:
                raise ValueError("Voice simulation requires `async_mode=True`.")
            self._voice = _VoiceRun(
                config=voice_config,
                tts_model=voice_config.tts_model,
                stt_model=voice_config.stt_model,
            )
            if voice_config.interruption_settings is not None:
                warnings.warn(
                    "`VoiceConfig.interruption_settings` is deprecated; set "
                    "`Persona(interruption_behavior=...)` on the golden "
                    "instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self.model_callback = model_callback
        self.is_callback_async = inspect.iscoroutinefunction(
            self.model_callback
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.async_mode = async_mode
        self.language = language
        self.simulated_conversations: List[ConversationalTestCase] = []
        self.simulator_model, self.using_native_model = initialize_model(
            simulator_model
        )
        # `None` is rewritten to the default node so the runtime path is
        # uniform: `_SimulationGraphRunner` always drives user-turn generation.
        # To customize the prompt template, pass
        # `simulation_graph=default_simulation_node(template=MyTemplate)`.
        self.simulation_graph = (
            simulation_graph
            if simulation_graph is not None
            else default_simulation_node()
        )
        self._graph_runner = _SimulationGraphRunner(root=self.simulation_graph)
        self.stopping_controller = SimulationController(
            controller=stopping_controller,
            generate_schema=self.generate_schema,
            a_generate_schema=self.a_generate_schema,
        )

    @property
    def tts_cost(self) -> float:
        """Speech synthesis spend for the last run; zero outside voice mode."""
        return self._voice.tts_cost if self._voice is not None else 0.0

    @property
    def stt_cost(self) -> float:
        """Transcription spend for the last run; zero outside voice mode."""
        return self._voice.stt_cost if self._voice is not None else 0.0

    @property
    def voice_config(self) -> Optional["VoiceConfig"]:
        return self._voice.config if self._voice is not None else None

    @property
    def voice_connector(self) -> Optional["BaseVoiceConnector"]:
        if self._voice is None or callable(self._voice.config.connector):
            return None
        return self._voice.config.connector

    def simulate(
        self,
        conversational_goldens: List[ConversationalGolden],
        max_user_simulations: int = 10,
        on_simulation_complete: Optional[
            Callable[[ConversationalTestCase, int], None]
        ] = None,
        on_turn: Optional[OnTurnCallback] = None,
    ) -> List[ConversationalTestCase]:
        self.simulation_cost = 0 if self.using_native_model else None
        if self._voice is not None:
            self._voice.begin_run(len(conversational_goldens))

        with conversation_simulator_progress_context(
            simulator_model=self.simulator_model.get_model_name(),
            num_conversations=len(conversational_goldens),
            async_mode=self.async_mode,
        ) as (progress, pbar_id), progress:

            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self._a_simulate(
                        conversational_goldens=conversational_goldens,
                        max_user_simulations=max_user_simulations,
                        on_simulation_complete=on_simulation_complete,
                        on_turn=on_turn,
                        progress=progress,
                        pbar_id=pbar_id,
                    )
                )
            else:
                multimodal = any(
                    [golden.multimodal for golden in conversational_goldens]
                )
                if multimodal:
                    if (
                        not self.simulator_model
                        or not self.simulator_model.supports_multimodal()
                    ):
                        if (
                            self.simulator_model
                            and type(self.simulator_model)
                            in MULTIMODAL_SUPPORTED_MODELS
                        ):
                            raise ValueError(
                                f"The evaluation model {self.simulator_model.name} does not support multimodal evaluations at the moment. Available multi-modal models for the {self.simulator_model.__class__.__name__} provider includes {', '.join(self.simulator_model.__class__.valid_multimodal_models)}."
                            )
                        else:
                            raise ValueError(
                                f"The evaluation model {self.simulator_model.name} does not support multimodal inputs, please use one of the following evaluation models: {', '.join([cls.__name__ for cls in MULTIMODAL_SUPPORTED_MODELS])}"
                            )
                conversational_test_cases: List[ConversationalTestCase] = []
                for conversation_index, golden in enumerate(
                    conversational_goldens
                ):
                    conversational_test_case = (
                        self._simulate_single_conversation(
                            golden=golden,
                            max_user_simulations=max_user_simulations,
                            index=conversation_index,
                            progress=progress,
                            pbar_id=pbar_id,
                            on_simulation_complete=on_simulation_complete,
                            on_turn=on_turn,
                        )
                    )
                    conversational_test_cases.append(conversational_test_case)

                self.simulated_conversations = conversational_test_cases

        return self.simulated_conversations

    async def _a_simulate(
        self,
        conversational_goldens: List[ConversationalGolden],
        max_user_simulations: int,
        on_simulation_complete: Optional[
            Callable[[ConversationalTestCase, int], None]
        ] = None,
        on_turn: Optional[OnTurnCallback] = None,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ) -> List[ConversationalTestCase]:

        multimodal = any(
            [golden.multimodal for golden in conversational_goldens]
        )
        if multimodal:
            if (
                not self.simulator_model
                or not self.simulator_model.supports_multimodal()
            ):
                if (
                    self.simulator_model
                    and type(self.simulator_model)
                    in MULTIMODAL_SUPPORTED_MODELS
                ):
                    raise ValueError(
                        f"The evaluation model {self.simulator_model.name} does not support multimodal evaluations at the moment. Available multi-modal models for the {self.simulator_model.__class__.__name__} provider includes {', '.join(self.simulator_model.__class__.valid_multimodal_models)}."
                    )
                else:
                    raise ValueError(
                        f"The evaluation model {self.simulator_model.name} does not support multimodal inputs, please use one of the following evaluation models: {', '.join([cls.__name__ for cls in MULTIMODAL_SUPPORTED_MODELS])}"
                    )

        self.simulation_cost = 0 if self.using_native_model else None

        async def simulate_conversations(
            golden: ConversationalGolden,
            conversation_index: int,
        ):
            async with self.semaphore:
                return await self._a_simulate_single_conversation(
                    golden=golden,
                    max_user_simulations=max_user_simulations,
                    index=conversation_index,
                    progress=progress,
                    pbar_id=pbar_id,
                    on_simulation_complete=on_simulation_complete,
                    on_turn=on_turn,
                )

        tasks = [
            simulate_conversations(golden, i)
            for i, golden in enumerate(conversational_goldens)
        ]
        self.simulated_conversations = await asyncio.gather(*tasks)

    ############################################
    ### Simulate Single Conversation ###########
    ############################################

    def _simulate_single_conversation(
        self,
        golden: ConversationalGolden,
        max_user_simulations: int,
        index: int,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
        on_simulation_complete: Optional[
            Callable[[ConversationalTestCase, int], None]
        ] = None,
        on_turn: Optional[OnTurnCallback] = None,
    ) -> ConversationalTestCase:
        simulation_counter = 0
        if max_user_simulations <= 0:
            raise ValueError("max_user_simulations must be greater than 0")

        # Define pbar
        pbar_max_user_simluations_id = add_pbar(
            progress,
            f"\t⚡ Test case #{index}",
            total=max_user_simulations + 1,
        )

        additional_metadata = {
            "Persona": (
                golden.persona.characteristics
                if golden.persona is not None
                else None
            )
        }
        user_input = None
        thread_id = str(uuid.uuid4())
        turns: List[Turn] = []
        graph_state: _GraphConversationState = (
            self._graph_runner.new_conversation_state()
        )

        if golden.turns is not None:
            turns.extend(golden.turns)

        while True:
            if simulation_counter >= max_user_simulations:
                update_pbar(progress, pbar_max_user_simluations_id)
                break

            # Stop conversation if needed
            should_stop_simulation = self.stopping_controller.run(
                turns=turns,
                golden=golden,
                index=index,
                thread_id=thread_id,
                simulation_counter=simulation_counter,
                max_user_simulations=max_user_simulations,
                progress=progress,
                pbar_turns_id=pbar_max_user_simluations_id,
            )
            if should_stop_simulation:
                break

            # Generate turn from user (via simulation graph)
            emission_end = False
            if len(turns) > 0 and turns[-1].role == "user":
                user_input = turns[-1].content
            else:
                emission = self._graph_runner.run(
                    self,
                    graph_state,
                    turns,
                    golden,
                    thread_id,
                    self.language,
                )
                emission_end = emission.end
                if emission.turn is None:
                    # max_visits exhausted on entry; end without another turn.
                    update_pbar(progress, pbar_max_user_simluations_id)
                    break
                turns.append(emission.turn)
                user_input = emission.turn.content
                update_pbar(progress, pbar_max_user_simluations_id)
                simulation_counter += 1
                _dispatch_on_turn_sync(on_turn, turns, index)

            # Generate turn from assistant
            if self.is_callback_async:
                assistant_turn = asyncio.run(
                    self.a_generate_turn_from_callback(
                        user_input,
                        model_callback=self.model_callback,
                        turns=turns,
                        thread_id=thread_id,
                    )
                )
            else:
                assistant_turn = self.generate_turn_from_callback(
                    user_input,
                    model_callback=self.model_callback,
                    turns=turns,
                    thread_id=thread_id,
                )
            turns.append(assistant_turn)
            _dispatch_on_turn_sync(on_turn, turns, index)

            # Route to the next graph node based on the assistant reply.
            self._graph_runner.advance(
                self, graph_state, assistant_turn.content
            )

            if emission_end:
                break

        update_pbar(progress, pbar_id)
        conversational_test_case = ConversationalTestCase(
            turns=turns,
            scenario=golden.scenario,
            expected_outcome=golden.expected_outcome,
            user_description=golden.user_description,
            context=golden.context,
            name=golden.name,
            additional_metadata={
                **(golden.additional_metadata or {}),
                **additional_metadata,
            },
            comments=golden.comments,
            _dataset_rank=golden._dataset_rank,
            _dataset_alias=golden._dataset_alias,
            _dataset_id=golden._dataset_id,
        )
        if on_simulation_complete:
            on_simulation_complete(conversational_test_case, index)
        return conversational_test_case

    async def _a_simulate_single_conversation(
        self,
        golden: ConversationalGolden,
        max_user_simulations: int,
        index: Optional[int] = None,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
        on_simulation_complete: Optional[
            Callable[[ConversationalTestCase, int], None]
        ] = None,
        on_turn: Optional[OnTurnCallback] = None,
    ) -> ConversationalTestCase:
        simulation_counter = 0
        if max_user_simulations <= 0:
            raise ValueError("max_user_simulations must be greater than 0")

        # Define pbar
        pbar_max_user_simluations_id = add_pbar(
            progress,
            f"\t⚡ Test case #{index}",
            total=max_user_simulations + 1,
        )

        additional_metadata = {
            "Persona": (
                golden.persona.characteristics
                if golden.persona is not None
                else None
            )
        }
        user_input = None
        thread_id = str(uuid.uuid4())
        turns: List[Turn] = []
        graph_state: _GraphConversationState = (
            self._graph_runner.new_conversation_state()
        )

        if golden.turns is not None:
            turns.extend(golden.turns)

        voice = self._voice
        session: Optional[_VoiceSession] = None
        if voice is not None:
            policy, floor = self._build_interruption(golden)
            session = voice.open_session(golden.persona, policy, floor)

        async with AsyncExitStack() as stack:
            # Voice mode: one live session per conversation. The connector is
            # connected before the first turn and disconnected when the
            # conversation ends (or errors).
            if session is not None:
                connect_started = time.perf_counter()
                logger.debug(
                    "Voice connector connect started: %s",
                    type(session.connector).__name__,
                )
                await stack.enter_async_context(session.connector)
                session.started_at = time.perf_counter()
                logger.debug(
                    "Voice connector connected after %.2fs",
                    time.perf_counter() - connect_started,
                )

            persona = golden.persona
            voice_mode = session is not None
            muted = voice_mode and persona is not None and persona.muted
            hold_timeout = persona.hold_timeout if persona is not None else None
            silent_for = 0.0

            if voice_mode and persona is not None and not persona.speaks_first:
                logger.debug("Persona waits to speak; listening for greeting")
                if session.is_duplex:
                    await self._voice_duplex_listen(session, turns, golden)
                else:
                    turns.append(await self._voice_listen(session, turns))
                await _dispatch_on_turn(on_turn, turns, index)

            while True:
                logger.debug(
                    "Simulation loop started: conversation=%s user_turn=%d/%d total_turns=%d",
                    index,
                    simulation_counter + 1,
                    max_user_simulations,
                    len(turns),
                )
                if simulation_counter >= max_user_simulations:
                    logger.debug("Maximum user simulations reached")
                    update_pbar(progress, pbar_max_user_simluations_id)
                    break

                # In voice mode the wait before the simulated user speaks is
                # audible to the agent, which may fill it, re-prompt, or hang
                # up — so the harness's own thinking time is not neutral, it
                # changes the behavior under test. The stopping check and the
                # next turn both read the conversation so far and neither uses
                # the other's result, so they run together and the turn is
                # discarded if the check says to stop. Outside voice mode the
                # delay costs only wall time, which is not worth a generation
                # thrown away on every conversation.
                emission_task: Optional[asyncio.Task] = None
                speculating = (
                    voice_mode
                    and not muted
                    and not (turns and turns[-1].role == "user")
                )
                if speculating:
                    user_started = time.perf_counter()
                    logger.debug("Simulated user generation started")
                    emission_task = asyncio.create_task(
                        self._graph_runner.a_run(
                            self,
                            graph_state,
                            turns,
                            golden,
                            thread_id,
                            self.language,
                        )
                    )

                # Stop conversation if needed
                controller_started = time.perf_counter()
                logger.debug("Stopping controller started")
                should_stop_simulation = await self.stopping_controller.a_run(
                    turns=turns,
                    golden=golden,
                    index=index if index is not None else 0,
                    thread_id=thread_id,
                    simulation_counter=simulation_counter,
                    max_user_simulations=max_user_simulations,
                    progress=progress,
                    pbar_turns_id=pbar_max_user_simluations_id,
                )
                logger.debug(
                    "Stopping controller finished after %.2fs: should_stop=%s",
                    time.perf_counter() - controller_started,
                    should_stop_simulation,
                )
                if should_stop_simulation:
                    await _discard_task(emission_task)
                    break

                # Generate turn from user (via simulation graph)
                emission_end = False
                if len(turns) > 0 and turns[-1].role == "user":
                    user_input = turns[-1].content
                elif muted:
                    # A muted caller never speaks; the empty turn keeps the
                    # transcript alternating so the dead air is visible.
                    turns.append(Turn(role="user", content=""))
                    user_input = ""
                    update_pbar(progress, pbar_max_user_simluations_id)
                    simulation_counter += 1
                    await _dispatch_on_turn(on_turn, turns, index)
                else:
                    if emission_task is None:
                        user_started = time.perf_counter()
                        logger.debug("Simulated user generation started")
                        emission_task = asyncio.create_task(
                            self._graph_runner.a_run(
                                self,
                                graph_state,
                                turns,
                                golden,
                                thread_id,
                                self.language,
                            )
                        )
                    emission = await emission_task
                    logger.debug(
                        "Simulated user generation finished after %.2fs",
                        time.perf_counter() - user_started,
                    )
                    emission_end = emission.end
                    if emission.turn is None:
                        logger.debug("Simulation graph emitted no user turn")
                        update_pbar(progress, pbar_max_user_simluations_id)
                        break
                    turns.append(emission.turn)
                    user_input = emission.turn.content
                    logger.debug("Simulated user turn: %r", user_input)
                    update_pbar(progress, pbar_max_user_simluations_id)
                    simulation_counter += 1
                    await _dispatch_on_turn(on_turn, turns, index)

                # Generate turn from assistant (half-duplex or duplex barge-in)
                assistant_started = time.perf_counter()
                logger.debug("Assistant voice exchange started")
                if muted:
                    assistant_turn = await self._voice_listen(session, turns)
                    turns.append(assistant_turn)
                elif session is not None and session.is_duplex:
                    user_turns_before = sum(
                        1 for turn in turns if turn.role == "user"
                    )
                    assistant_turn = await self._voice_duplex_exchange(
                        session, user_input, turns, golden
                    )
                    simulation_counter += (
                        sum(1 for turn in turns if turn.role == "user")
                        - user_turns_before
                    )
                elif session is not None:
                    assistant_turn = await self._voice_model_callback(
                        session, user_input, turns
                    )
                    turns.append(assistant_turn)
                elif self.is_callback_async:
                    assistant_turn = await self.a_generate_turn_from_callback(
                        user_input,
                        model_callback=self.model_callback,
                        turns=turns,
                        thread_id=thread_id,
                    )
                    turns.append(assistant_turn)
                else:
                    assistant_turn = self.generate_turn_from_callback(
                        user_input,
                        model_callback=self.model_callback,
                        turns=turns,
                        thread_id=thread_id,
                    )
                    turns.append(assistant_turn)

                await _dispatch_on_turn(on_turn, turns, index)

                exchange_seconds = time.perf_counter() - assistant_started
                logger.debug(
                    "Assistant voice exchange finished after %.2fs: transcript=%r latency_ms=%s",
                    exchange_seconds,
                    assistant_turn.content,
                    assistant_turn.latency_ms,
                )

                # Hang up on hold music or dead air rather than waiting out
                # the agent's own (usually much longer) timeout.
                if hold_timeout is not None:
                    if assistant_turn.content.strip():
                        silent_for = 0.0
                    else:
                        silent_for += exchange_seconds
                        if silent_for >= hold_timeout:
                            logger.debug(
                                "Hanging up after %.2fs without agent speech "
                                "(hold_timeout=%.2fs)",
                                silent_for,
                                hold_timeout,
                            )
                            break

                # Route to the next graph node based on the assistant reply.
                route_started = time.perf_counter()
                logger.debug("Simulation graph routing started")
                await self._graph_runner.a_advance(
                    self, graph_state, assistant_turn.content
                )
                logger.debug(
                    "Simulation graph routing finished after %.2fs",
                    time.perf_counter() - route_started,
                )

                if emission_end:
                    break

        update_pbar(progress, pbar_id)
        if voice is not None:
            self._log_voice_latency(turns, index)
        call_recording_path = (
            session.recorder.finish()
            if session is not None and session.recorder is not None
            else None
        )
        conversational_test_case = ConversationalTestCase(
            turns=turns,
            scenario=golden.scenario,
            expected_outcome=golden.expected_outcome,
            user_description=golden.user_description,
            context=golden.context,
            name=golden.name,
            additional_metadata={
                **(golden.additional_metadata or {}),
                **additional_metadata,
            },
            comments=golden.comments,
            _dataset_rank=golden._dataset_rank,
            _dataset_alias=golden._dataset_alias,
            _dataset_id=golden._dataset_id,
        )
        conversational_test_case.call_recording_path = call_recording_path
        if voice is not None and voice.config.output_dir is not None:
            save_started = time.perf_counter()
            logger.debug("Saving voice audio to %s", voice.config.output_dir)
            self._save_voice_audio(conversational_test_case, golden, index)
            logger.debug(
                "Voice audio saved after %.2fs",
                time.perf_counter() - save_started,
            )
        if on_simulation_complete:
            on_simulation_complete(conversational_test_case, index)
        return conversational_test_case

    @staticmethod
    def _persona_tts_kwargs(persona) -> dict:
        return {} if persona is None else persona.tts_kwargs()

    @staticmethod
    def _stt_kwargs(session: _VoiceSession) -> dict:
        """Transcription kwargs for the active persona.

        `language="auto"` tells the STT model to detect the language per
        utterance instead of locking to the one it was configured with.
        """
        persona = session.persona
        if persona is not None and persona.multilingual_stt:
            return {"language": "auto"}
        return {}

    @staticmethod
    def _mix_background(session: _VoiceSession, audio: "Audio") -> "Audio":
        """Lay the persona's ambience under one uplink utterance."""
        persona = session.persona
        if persona is None or persona.background_noise is None:
            return audio
        from deepeval.voice.background import mix_background

        return mix_background(audio, persona.background_noise)

    async def _send_user_utterance(
        self,
        session: _VoiceSession,
        text: str,
        persona: Optional["Persona"],
        *,
        trailing_silence: bool,
    ) -> Tuple["Audio", float]:
        """Put the caller's next utterance on the uplink as it is synthesized.

        Returns the utterance and the moment it began going out, which only the
        transport can say: one that forwards frames sends the first while the
        rest is still being made, and one that needs the utterance whole cannot
        send anything until synthesis ends. Timing the clip from synthesis
        instead would place it on the call before the agent heard anything.
        """
        from deepeval.voice.background import BackgroundMixer

        voice = self._voice
        tts_kwargs = self._persona_tts_kwargs(persona)
        if not voice.tts_model.supports_streaming():
            audio, tts_cost = await voice.tts_model.a_synthesize(
                text, **tts_kwargs
            )
            if tts_cost is not None:
                voice.tts_cost += tts_cost
            _populate_audio_duration(audio)
            audio = self._mix_background(session, audio)
            started_at = time.perf_counter()
            await session.connector.stream_uplink(
                audio, trailing_silence=trailing_silence
            )
            return audio, started_at

        mixer = BackgroundMixer(
            persona.background_noise if persona is not None else None
        )

        async def _frames():
            async for chunk in voice.tts_model.a_synthesize_stream(
                text, **tts_kwargs
            ):
                yield mixer.mix_chunk(chunk)

        tts_started = time.perf_counter()
        logger.debug("User TTS stream started: characters=%d", len(text))
        result = await session.connector.stream_uplink_chunks(
            _frames(), trailing_silence=trailing_silence
        )
        tts_cost = voice.tts_model.synthesis_cost(text)
        if tts_cost is not None:
            voice.tts_cost += tts_cost
        logger.debug(
            "User TTS stream finished after %.2fs: sending began after %.2fs",
            time.perf_counter() - tts_started,
            (result.first_frame_at or tts_started) - tts_started,
        )
        return result.audio, result.first_frame_at or time.perf_counter()

    @staticmethod
    def _silence_audio(session: _VoiceSession, seconds: float) -> "Audio":
        """Digital silence in the connector's uplink format."""
        from deepeval.test_case import Audio
        from deepeval.voice.connectors import audio_utils

        sample_rate, _ = session.connector.audio_format
        pcm = b"\x00" * (int(sample_rate * seconds) * 2)
        return Audio.from_bytes(
            audio_utils.pcm16_to_wav_bytes(pcm, sample_rate, 1),
            "audio/wav",
            sampleRate=sample_rate,
            encoding="wav",
            duration=seconds,
        )

    @staticmethod
    def _log_voice_latency(turns: List[Turn], index: int) -> None:
        """Log each side's reply time; the caller's is the simulator's own."""
        from deepeval.voice.timeline import build_audio_timeline

        agent_waits = [
            turn.latency_ms / 1000.0
            for turn in turns
            if turn.role == "assistant" and turn.latency_ms is not None
        ]
        timeline = build_audio_timeline(turns)
        caller_waits = [
            following.start_time - current.end_time
            for current, following in zip(timeline, timeline[1:])
            if current.role == "assistant" and following.role == "user"
        ]
        if not agent_waits and not caller_waits:
            return

        def describe(waits: List[float]) -> str:
            if not waits:
                return "n/a"
            return "avg %.2fs, max %.2fs over %d turns" % (
                sum(waits) / len(waits),
                max(waits),
                len(waits),
            )

        logger.info(
            "Conversation %d voice latency: agent replied after %s; "
            "caller replied after %s",
            index,
            describe(agent_waits),
            describe(caller_waits),
        )

    async def _voice_listen(
        self, session: _VoiceSession, turns: List[Turn]
    ) -> Turn:
        """Say nothing and record whatever the agent says back."""
        silence = self._mix_background(
            session, self._silence_audio(session, _SILENCE_PROBE_SECONDS)
        )
        if turns and turns[-1].role == "user":
            turns[-1].audio = silence
        return await self._voice_exchange(session, silence, turns)

    def _build_interruption(self, golden: ConversationalGolden):
        """Build this conversation's barge-in policy and floor controller.

        The golden's persona wins; `VoiceConfig.interruption_settings` is the
        deprecated run-wide fallback. Returns `(None, None)` when nobody
        barges; a full-duplex transport still runs the duplex exchange then.
        """
        from deepeval.voice.floor_control import FloorController
        from deepeval.voice.interruption import interruption_policy

        behavior = (
            golden.persona.interruption_behavior
            if golden.persona is not None
            else None
        )
        if behavior is None:
            behavior = self._voice.config.interruption_settings
        if behavior is None:
            return None, None

        policy = interruption_policy(behavior.frequency)
        return policy, FloorController.from_overlap_behavior(
            behavior.overlap, policy=policy
        )

    def _save_voice_audio(
        self,
        test_case: ConversationalTestCase,
        golden: ConversationalGolden,
        index: Optional[int],
    ) -> None:
        from deepeval.voice.output import save_conversation_audio

        voice = self._voice
        conversation_id = None
        if max(voice.num_goldens, 1) > 1:
            conversation_id = (
                golden.name
                or f"conversation-{index if index is not None else 0}"
            )
        save_conversation_audio(
            test_case,
            output_dir=voice.config.output_dir,
            run_label=voice.run_label,
            conversation_id=conversation_id,
            combine_audio_files=voice.config.combine_audio_files,
        )

    ############################################
    ### Generate User Inputs ###################
    ############################################

    def generate_first_user_input(
        self,
        golden: ConversationalGolden,
        template: Optional[Type[SimulationTemplate]] = None,
    ):
        tmpl = template or SimulationTemplate
        prompt = tmpl.simulate_first_user_turn(golden, self.language)
        simulated_input: SimulatedInput = self.generate_schema(
            prompt, SimulatedInput
        )
        return simulated_input.simulated_input

    async def a_generate_first_user_input(
        self,
        golden: ConversationalGolden,
        template: Optional[Type[SimulationTemplate]] = None,
    ):
        tmpl = template or SimulationTemplate
        prompt = tmpl.simulate_first_user_turn(golden, self.language)
        simulated_input: SimulatedInput = await self.a_generate_schema(
            prompt, SimulatedInput
        )
        return simulated_input.simulated_input

    def generate_next_user_input(
        self,
        golden: ConversationalGolden,
        turns: List[Turn],
        template: Optional[Type[SimulationTemplate]] = None,
    ):
        tmpl = template or SimulationTemplate
        prompt = tmpl.simulate_user_turn(golden, turns, self.language)
        simulated_input: SimulatedInput = self.generate_schema(
            prompt, SimulatedInput
        )
        return simulated_input.simulated_input

    async def a_generate_next_user_input(
        self,
        golden: ConversationalGolden,
        turns: List[Turn],
        template: Optional[Type[SimulationTemplate]] = None,
    ):
        tmpl = template or SimulationTemplate
        prompt = tmpl.simulate_user_turn(golden, turns, self.language)
        simulated_input: SimulatedInput = await self.a_generate_schema(
            prompt, SimulatedInput
        )
        return simulated_input.simulated_input

    ############################################
    ### Generate Structured Response ###########
    ############################################

    def generate_schema(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:
        if self.using_native_model:
            res, cost = self.simulator_model.generate(prompt, schema=schema)
            if cost is not None:
                self.simulation_cost += cost
            return res
        else:
            try:
                res = self.simulator_model.generate(prompt, schema=schema)
                return res
            except TypeError:
                res = self.simulator_model.generate(prompt)
                data = trimAndLoadJson(res)
                return schema(**data)

    async def a_generate_schema(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:
        if self.using_native_model:
            res, cost = await self.simulator_model.a_generate(
                prompt, schema=schema
            )
            if cost is not None:
                self.simulation_cost += cost
            return res
        else:
            try:
                res = await self.simulator_model.a_generate(
                    prompt, schema=schema
                )
                return res
            except TypeError:
                res = await self.simulator_model.a_generate(prompt)
            data = trimAndLoadJson(res)
            return schema(**data)

    ############################################
    ### Invoke Model Callback ##################
    ############################################

    def generate_turn_from_callback(
        self,
        input: str,
        turns: List[Turn],
        thread_id: str,
        model_callback: Callable,
    ) -> Turn:
        callback_kwargs = {
            "input": input,
            "turns": turns,
            "thread_id": thread_id,
        }
        supported_args = set(
            inspect.signature(model_callback).parameters.keys()
        )
        return model_callback(
            **{k: v for k, v in callback_kwargs.items() if k in supported_args}
        )

    async def a_generate_turn_from_callback(
        self,
        input: str,
        model_callback: Callable,
        turns: List[Turn],
        thread_id: str,
    ) -> Turn:
        candidate_kwargs = {
            "input": input,
            "turns": turns,
            "thread_id": thread_id,
        }
        supported_args = set(
            inspect.signature(model_callback).parameters.keys()
        )
        return await model_callback(
            **{k: v for k, v in candidate_kwargs.items() if k in supported_args}
        )

    async def _voice_model_callback(
        self, session: _VoiceSession, input: str, turns: List[Turn]
    ) -> Turn:
        """Half-duplex voice callback: TTS → exchange_turn → STT.

        The fallback for transports that cannot carry both voices at once;
        everything else goes through `_voice_duplex_exchange`.
        """
        voice = self._voice
        tts_started = time.perf_counter()
        logger.debug("User TTS started: characters=%d", len(input))
        user_audio, tts_cost = await voice.tts_model.a_synthesize(
            input, **self._persona_tts_kwargs(session.persona)
        )
        _populate_audio_duration(user_audio)
        logger.debug(
            "User TTS finished after %.2fs: bytes=%d",
            time.perf_counter() - tts_started,
            len(user_audio.get_bytes()),
        )
        if tts_cost is not None:
            voice.tts_cost += tts_cost
        user_audio = self._mix_background(session, user_audio)
        if turns and turns[-1].role == "user":
            turns[-1].audio = user_audio

        return await self._voice_exchange(session, user_audio, turns)

    async def _voice_exchange(
        self, session: _VoiceSession, user_audio: "Audio", turns: List[Turn]
    ) -> Turn:
        """Play `user_audio` to the agent and transcribe the reply.

        Shared by the half-duplex callback and by silence-only exchanges (a
        muted caller, or waiting out an agent that speaks first).
        """
        voice = self._voice
        connector_started = time.perf_counter()
        logger.debug(
            "Connector exchange started: %s",
            type(session.connector).__name__,
        )
        conn_turn = await session.connector.exchange_turn(user_audio)
        _populate_audio_duration(conn_turn.audio)
        call_started_at = session.started_at or connector_started
        user_audio.start_time = max(
            0.0,
            (conn_turn.input_audio_started_at or connector_started)
            - call_started_at,
        )
        assistant_audio_started_at = conn_turn.audio_started_at
        if assistant_audio_started_at is None:
            input_ended_at = (
                conn_turn.input_audio_ended_at or time.perf_counter()
            )
            if conn_turn.latency_ms is not None:
                assistant_audio_started_at = (
                    input_ended_at + conn_turn.latency_ms / 1000.0
                )
        if (
            assistant_audio_started_at is not None
            and conn_turn.audio is not None
        ):
            conn_turn.audio.start_time = max(
                0.0, assistant_audio_started_at - call_started_at
            )
        logger.debug(
            "Connector exchange finished after %.2fs: transcript=%s latency_ms=%s",
            time.perf_counter() - connector_started,
            bool(conn_turn.transcript),
            conn_turn.latency_ms,
        )
        has_audio = (
            conn_turn.audio is not None
            and len(conn_turn.audio.get_bytes()) > 44  # more than a WAV header
            and _has_speech_energy(conn_turn.audio, session.connector)
        )
        if conn_turn.transcript:
            agent_text = conn_turn.transcript
            logger.debug("Connector transcript supplied; STT skipped")
        elif has_audio:
            stt_started = time.perf_counter()
            logger.debug("Agent STT started")
            agent_text, stt_cost = await voice.stt_model.a_transcribe(
                conn_turn.audio, **self._stt_kwargs(session)
            )
            logger.debug(
                "Agent STT finished after %.2fs",
                time.perf_counter() - stt_started,
            )
            if stt_cost is not None:
                voice.stt_cost += stt_cost
        else:
            agent_text = ""
            logger.warning(
                "Turn %d produced no agent transcript or audio; the agent may "
                "not be responding (check connection, credentials, and audio "
                "format).",
                len(turns),
            )

        # Half-duplex: leave Turn.interrupted unset. Provider interruption
        # events are often spurious when the next user turn arrives while
        # the previous reply is still playing server-side.
        replied = has_audio or bool(conn_turn.transcript)
        return Turn(
            role="assistant",
            content=agent_text,
            audio=conn_turn.audio if replied else None,
            latency_ms=conn_turn.latency_ms if replied else None,
        )

    async def _voice_duplex_exchange(
        self,
        session: _VoiceSession,
        input: str,
        turns: List[Turn],
        golden: ConversationalGolden,
    ) -> Turn:
        """Duplex voice exchange with mid-speech barge-in and floor control.

        Appends barge user turns and assistant turns onto `turns` in place.
        Returns the last assistant turn for simulation-graph routing.
        """
        call_started_at = session.started_at or time.perf_counter()
        user_audio, uplink_started_at = await self._send_user_utterance(
            session, input, golden.persona, trailing_silence=True
        )
        user_audio.start_time = max(0.0, uplink_started_at - call_started_at)
        if turns and turns[-1].role == "user":
            turns[-1].audio = user_audio
        # Latency is the agent's wait, which starts when the caller stops
        # speaking. Deriving that from the utterance keeps it comparable across
        # connectors, whose uplink calls return at wildly different points —
        # in-process ones return before a single frame has been heard.
        sent_at = uplink_started_at + (user_audio.duration or 0.0)

        assistant_turn = await self._run_duplex(session, golden, turns, sent_at)
        if assistant_turn is not None:
            return assistant_turn
        logger.warning(
            "Duplex exchange produced no assistant turn; inserting empty reply."
        )
        empty = Turn(role="assistant", content="")
        turns.append(empty)
        return empty

    async def _run_duplex(
        self,
        session: _VoiceSession,
        golden: ConversationalGolden,
        turns: List[Turn],
        sent_at: float,
    ) -> Optional[Turn]:
        """Listen until the agent's turn ends; None if it never spoke."""
        from deepeval.voice.duplex import DuplexExchange

        voice = self._voice
        exchange = DuplexExchange(
            connector=session.connector,
            tts_model=voice.tts_model,
            stt_model=voice.stt_model,
            policy=session.policy,
            floor=session.floor,
            golden=golden,
            language=self.language,
            a_generate_schema=self.a_generate_schema,
            call_started_at=session.started_at or time.perf_counter(),
        )
        result = await exchange.run(
            turns=turns,
            sent_at=sent_at,
            barges_this_conversation=session.barges,
        )
        session.barges += result.barges
        voice.tts_cost += result.tts_cost
        voice.stt_cost += result.stt_cost
        return next(
            (
                turn
                for turn in reversed(result.turns)
                if turn.role == "assistant"
            ),
            None,
        )

    async def _voice_duplex_listen(
        self,
        session: _VoiceSession,
        turns: List[Turn],
        golden: ConversationalGolden,
    ) -> None:
        """Wait out the greeting, or give the caller the floor if none comes."""
        persona = session.persona
        timeout = (
            persona.hold_timeout
            if persona is not None and persona.hold_timeout
            else _GREETING_TIMEOUT_S
        )
        try:
            await asyncio.wait_for(
                self._run_duplex(session, golden, turns, time.perf_counter()),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.debug("No greeting heard; the caller speaks first")

    ############################################
    ### Invoke Model Callback ##################
    ############################################
