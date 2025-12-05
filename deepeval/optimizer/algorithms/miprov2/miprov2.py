# MIPROv2 - Multiprompt Instruction PRoposal Optimizer Version 2
#
# This implementation follows the original MIPROv2 paper and DSPy implementation:
# https://arxiv.org/pdf/2406.11695
# https://dspy.ai/api/optimizers/MIPROv2/
#
# The algorithm works in two phases:
#
#   1. PROPOSAL PHASE:
#      a) Generate N diverse instruction candidates upfront
#      b) Bootstrap few-shot demonstration sets from training data
#
#   2. OPTIMIZATION PHASE: Use Bayesian Optimization (Optuna TPE) to search
#      over the joint space of (instruction_candidate, demo_set). Each trial:
#      - Samples an instruction candidate index
#      - Samples a demo set index
#      - Renders the prompt with demos
#      - Evaluates on a minibatch of examples
#      - Uses the score to guide the Bayesian surrogate model
#
# Periodic full evaluation is performed every `minibatch_full_eval_steps`
# to get accurate scores on the complete validation set.


from __future__ import annotations
import asyncio
import uuid
import random
import time
import logging
from typing import (
    Dict,
    List,
    Tuple,
    TYPE_CHECKING,
    Union,
    Optional,
    Callable,
)

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.errors import DeepEvalError
from deepeval.optimizer.utils import Aggregator, mean_of_all
from deepeval.optimizer.types import (
    PromptConfiguration,
    PromptConfigurationId,
    ModuleId,
    ScoreTable,
    OptimizationReport,
    RunnerStatusType,
    RunnerStatusCallback,
)
from deepeval.optimizer.scorer.base import BaseScorer
from deepeval.optimizer.algorithms.base import BaseAlgorithm
from deepeval.optimizer.utils import build_prompt_config_snapshots
from deepeval.prompt.prompt import Prompt
from deepeval.optimizer.algorithms.miprov2.proposer import InstructionProposer
from deepeval.optimizer.algorithms.miprov2.bootstrapper import (
    DemoBootstrapper,
    DemoSet,
    render_prompt_with_demos,
)
from deepeval.optimizer.algorithms.configs import (
    MIPROV2_DEFAULT_NUM_CANDIDATES,
    MIPROV2_DEFAULT_NUM_TRIALS,
    MIPROV2_DEFAULT_MINIBATCH_SIZE,
    MIPROV2_DEFAULT_MINIBATCH_FULL_EVAL_STEPS,
    MIPROV2_DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
    MIPROV2_DEFAULT_MAX_LABELED_DEMOS,
    MIPROV2_DEFAULT_NUM_DEMO_SETS,
)

if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


# Suppress Optuna's verbose logging
logging.getLogger("optuna").setLevel(logging.WARNING)


class MIPROV2(BaseAlgorithm):
    """
    MIPROv2 (Multiprompt Instruction PRoposal Optimizer Version 2)

    A prompt optimizer that uses Bayesian Optimization to find the best
    combination of instruction and few-shot demonstrations. Follows the
    original MIPROv2 paper approach.

    The optimization process:
    1. Generate N diverse instruction candidates upfront
    2. Bootstrap M demo sets from training examples
    3. Use Optuna's TPE sampler for Bayesian Optimization over (instruction, demos)
    4. Each trial evaluates a combination on a minibatch
    5. Periodically evaluate the best combination on the full dataset

    Parameters
    ----------
    num_candidates : int
        Number of instruction candidates to propose. Default is 10.
    num_trials : int
        Number of Bayesian Optimization trials. Default is 20.
    minibatch_size : int
        Number of examples per minibatch evaluation. Default is 25.
    minibatch_full_eval_steps : int
        Evaluate best on full dataset every N trials. Default is 10.
    max_bootstrapped_demos : int
        Maximum bootstrapped demos per demo set. Default is 4.
    max_labeled_demos : int
        Maximum labeled demos (from expected_output) per set. Default is 4.
    num_demo_sets : int
        Number of demo sets to create. Default is 5.
    random_seed : int, optional
        RNG seed for reproducibility. If None, derived from time.time_ns().
    aggregate_instances : Aggregator
        Function to aggregate per-instance scores. Default is mean_of_all.
    scorer : BaseScorer, optional
        Scorer for evaluating prompts. Set by PromptOptimizer.
    """

    name = "MIPROv2"
    SINGLE_MODULE_ID: ModuleId = "__module__"

    def __init__(
        self,
        num_candidates: int = MIPROV2_DEFAULT_NUM_CANDIDATES,
        num_trials: int = MIPROV2_DEFAULT_NUM_TRIALS,
        minibatch_size: int = MIPROV2_DEFAULT_MINIBATCH_SIZE,
        minibatch_full_eval_steps: int = MIPROV2_DEFAULT_MINIBATCH_FULL_EVAL_STEPS,
        max_bootstrapped_demos: int = MIPROV2_DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos: int = MIPROV2_DEFAULT_MAX_LABELED_DEMOS,
        num_demo_sets: int = MIPROV2_DEFAULT_NUM_DEMO_SETS,
        random_seed: Optional[int] = None,
        aggregate_instances: Aggregator = mean_of_all,
        scorer: Optional[BaseScorer] = None,
    ) -> None:
        if not OPTUNA_AVAILABLE:
            raise DeepEvalError(
                "MIPROv2 requires the 'optuna' package for Bayesian Optimization. "
                "Install it with: pip install optuna"
            )

        # Validate parameters
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if num_trials < 1:
            raise ValueError("num_trials must be >= 1")
        if minibatch_size < 1:
            raise ValueError("minibatch_size must be >= 1")
        if minibatch_full_eval_steps < 1:
            raise ValueError("minibatch_full_eval_steps must be >= 1")
        if max_bootstrapped_demos < 0:
            raise ValueError("max_bootstrapped_demos must be >= 0")
        if max_labeled_demos < 0:
            raise ValueError("max_labeled_demos must be >= 0")
        if num_demo_sets < 1:
            raise ValueError("num_demo_sets must be >= 1")

        self.num_candidates = num_candidates
        self.num_trials = num_trials
        self.minibatch_size = minibatch_size
        self.minibatch_full_eval_steps = minibatch_full_eval_steps
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.num_demo_sets = num_demo_sets
        self.aggregate_instances = aggregate_instances
        self.scorer = scorer

        # Random seed handling
        if random_seed is None:
            random_seed = time.time_ns() % (2**31)
        self.random_seed = random_seed
        self.random_state = random.Random(random_seed)

        # Runtime state
        self.reset_state()

        # Callbacks and models (set by PromptOptimizer)
        self.status_callback: Optional[RunnerStatusCallback] = None
        self.optimizer_model: Optional["DeepEvalBaseLLM"] = None

        # Lazy-loaded components
        self._proposer: Optional[InstructionProposer] = None
        self._bootstrapper: Optional[DemoBootstrapper] = None

    ##############
    # Public API #
    ##############

    def execute(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, OptimizationReport]:
        """
        Synchronous MIPROv2 optimization.

        Phase 1: Propose instruction candidates + Bootstrap demo sets
        Phase 2: Use Bayesian Optimization to find the best combination
        """
        self._validate_inputs(goldens)
        self._ensure_scorer()
        self._ensure_proposer()
        self._ensure_bootstrapper()
        self.reset_state()

        # Phase 1a: Propose instruction candidates
        self._update_status("Phase 1: Proposing instruction candidates...", 0)
        instruction_candidates = self._proposer.propose(
            prompt=prompt,
            goldens=goldens,
            num_candidates=self.num_candidates,
        )
        self._register_instruction_candidates(instruction_candidates)

        # Phase 1b: Bootstrap demo sets
        self._update_status(
            "Phase 1: Bootstrapping few-shot demonstrations...", 0
        )
        self._demo_sets = self._bootstrapper.bootstrap(
            prompt=prompt,
            goldens=goldens,
            generate_fn=self._create_generate_fn(),
        )
        self._update_status(f"Bootstrapped {len(self._demo_sets)} demo sets", 0)

        # Phase 2: Bayesian Optimization over (instruction, demos)
        self._update_status("Phase 2: Starting Bayesian Optimization...", 0)
        best_instr_idx, best_demo_idx = self._run_bayesian_optimization(goldens)

        # Final full evaluation if not already done
        config_key = (best_instr_idx, best_demo_idx)
        if config_key not in self._full_eval_cache:
            best_config = self._get_config_by_index(best_instr_idx)
            best_demo_set = self._demo_sets[best_demo_idx]
            self._full_evaluate(best_config, best_demo_set, goldens)

        # Build report
        best = self._best_by_aggregate()
        return self._build_result(best)

    async def a_execute(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, OptimizationReport]:
        """
        Asynchronous MIPROv2 optimization.
        """
        self._validate_inputs(goldens)
        self._ensure_scorer()
        self._ensure_proposer()
        self._ensure_bootstrapper()
        self.reset_state()

        # Phase 1: Run proposal and bootstrapping concurrently
        self._update_status(
            "Phase 1: Proposing candidates & bootstrapping demos...", 0
        )

        instruction_candidates, demo_sets = await asyncio.gather(
            self._proposer.a_propose(
                prompt=prompt,
                goldens=goldens,
                num_candidates=self.num_candidates,
            ),
            self._bootstrapper.a_bootstrap(
                prompt=prompt,
                goldens=goldens,
                a_generate_fn=self._create_async_generate_fn(),
            ),
        )

        self._register_instruction_candidates(instruction_candidates)
        self._demo_sets = demo_sets
        self._update_status(
            f"Generated {len(instruction_candidates)} candidates, {len(self._demo_sets)} demo sets",
            0,
        )

        # Phase 2: Bayesian Optimization
        self._update_status("Phase 2: Starting Bayesian Optimization...", 0)
        best_instr_idx, best_demo_idx = await self._a_run_bayesian_optimization(
            goldens
        )

        # Final full evaluation if not already done
        config_key = (best_instr_idx, best_demo_idx)
        if config_key not in self._full_eval_cache:
            best_config = self._get_config_by_index(best_instr_idx)
            best_demo_set = self._demo_sets[best_demo_idx]
            await self._a_full_evaluate(best_config, best_demo_set, goldens)

        # Build report
        best = self._best_by_aggregate()
        return self._build_result(best)

    ###################
    # State & Helpers #
    ###################

    def reset_state(self) -> None:
        """Reset optimization state for a new run."""
        self.optimization_id = str(uuid.uuid4())
        self.prompt_configurations_by_id: Dict[
            PromptConfigurationId, PromptConfiguration
        ] = {}
        self.parents_by_id: Dict[
            PromptConfigurationId, Optional[PromptConfigurationId]
        ] = {}
        self.pareto_score_table: ScoreTable = {}

        # Candidate tracking
        self._instruction_candidates: List[PromptConfiguration] = []
        self._demo_sets: List[DemoSet] = []

        # Score tracking: (instr_idx, demo_idx) -> list of minibatch scores
        self._combination_scores: Dict[Tuple[int, int], List[float]] = {}

        # Full eval cache: (instr_idx, demo_idx) -> config_id
        self._full_eval_cache: Dict[Tuple[int, int], PromptConfigurationId] = {}

        # Trial tracking
        self._trial_history: List[Dict] = []
        self._best_trial_key: Tuple[int, int] = (0, 0)
        self._best_trial_score: float = float("-inf")

    def _validate_inputs(
        self,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> None:
        """Validate input parameters."""
        if len(goldens) < 1:
            raise DeepEvalError(
                "MIPROv2 prompt optimization requires at least 1 golden, but "
                f"received {len(goldens)}. Provide at least one golden to run "
                "the optimizer."
            )

    def _ensure_scorer(self) -> None:
        """Ensure scorer is configured."""
        if self.scorer is None:
            raise DeepEvalError(
                "MIPROv2 requires a `scorer`. "
                "Construct one in PromptOptimizer and assign it to `runner.scorer`."
            )

    def _ensure_proposer(self) -> None:
        """Lazily initialize the instruction proposer."""
        if self._proposer is None:
            if self.optimizer_model is None:
                raise DeepEvalError(
                    "MIPROv2 requires an `optimizer_model` for instruction proposal. "
                    "Set it via PromptOptimizer."
                )
            self._proposer = InstructionProposer(
                optimizer_model=self.optimizer_model,
                random_state=self.random_state,
            )

    def _ensure_bootstrapper(self) -> None:
        """Lazily initialize the demo bootstrapper."""
        if self._bootstrapper is None:
            self._bootstrapper = DemoBootstrapper(
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                num_demo_sets=self.num_demo_sets,
                random_state=self.random_state,
            )

    def _create_generate_fn(
        self,
    ) -> Callable[[Prompt, Union["Golden", "ConversationalGolden"]], str]:
        """Create a sync generate function for bootstrapping."""

        def generate_fn(
            prompt: Prompt,
            golden: Union["Golden", "ConversationalGolden"],
        ) -> str:
            # Create a temporary config for generation
            temp_config = PromptConfiguration.new(
                prompts={self.SINGLE_MODULE_ID: prompt}
            )
            return self.scorer.generate(temp_config.prompts, golden)

        return generate_fn

    def _create_async_generate_fn(self) -> Callable:
        """Create an async generate function for bootstrapping."""

        async def a_generate_fn(
            prompt: Prompt,
            golden: Union["Golden", "ConversationalGolden"],
        ) -> str:
            temp_config = PromptConfiguration.new(
                prompts={self.SINGLE_MODULE_ID: prompt}
            )
            return await self.scorer.a_generate(temp_config.prompts, golden)

        return a_generate_fn

    def _register_instruction_candidates(
        self, candidates: List[Prompt]
    ) -> None:
        """Register all instruction candidates as configurations."""
        for i, prompt in enumerate(candidates):
            config = PromptConfiguration.new(
                prompts={self.SINGLE_MODULE_ID: prompt},
                parent=None if i == 0 else self._instruction_candidates[0].id,
            )
            self._instruction_candidates.append(config)
            self.prompt_configurations_by_id[config.id] = config
            self.parents_by_id[config.id] = config.parent

    def _get_config_by_index(self, idx: int) -> PromptConfiguration:
        """Get configuration by instruction candidate index."""
        return self._instruction_candidates[idx]

    def _draw_minibatch(
        self,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Union[List["Golden"], List["ConversationalGolden"]]:
        """Sample a minibatch from goldens."""
        n = len(goldens)
        if n <= 0:
            return []
        size = min(self.minibatch_size, n)
        return [goldens[self.random_state.randrange(0, n)] for _ in range(size)]

    def _render_config_with_demos(
        self,
        config: PromptConfiguration,
        demo_set: DemoSet,
    ) -> PromptConfiguration:
        """Create a new config with demos rendered into the prompt."""
        base_prompt = config.prompts[self.SINGLE_MODULE_ID]
        rendered_prompt = render_prompt_with_demos(
            prompt=base_prompt,
            demo_set=demo_set,
            max_demos=self.max_bootstrapped_demos + self.max_labeled_demos,
        )

        # Create a new config with the rendered prompt
        rendered_config = PromptConfiguration.new(
            prompts={self.SINGLE_MODULE_ID: rendered_prompt},
            parent=config.id,
        )
        return rendered_config

    ############################
    # Bayesian Optimization    #
    ############################

    def _run_bayesian_optimization(
        self,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[int, int]:
        """
        Run Bayesian Optimization using Optuna's TPE sampler.
        Returns the (instruction_idx, demo_set_idx) of the best combination.
        """
        num_instructions = len(self._instruction_candidates)
        num_demo_sets = len(self._demo_sets)

        # Create Optuna study with TPE sampler
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        def objective(trial: "optuna.Trial") -> float:
            # Sample instruction and demo set indices
            instr_idx = trial.suggest_int("instr_idx", 0, num_instructions - 1)
            demo_idx = trial.suggest_int("demo_idx", 0, num_demo_sets - 1)

            # Get the configuration and demo set
            config = self._get_config_by_index(instr_idx)
            demo_set = self._demo_sets[demo_idx]

            # Render prompt with demos
            rendered_config = self._render_config_with_demos(config, demo_set)

            # Draw minibatch and score
            minibatch = self._draw_minibatch(goldens)
            score = self.scorer.score_minibatch(rendered_config, minibatch)

            # Track scores for this combination
            combo_key = (instr_idx, demo_idx)
            if combo_key not in self._combination_scores:
                self._combination_scores[combo_key] = []
            self._combination_scores[combo_key].append(score)

            # Update best tracking
            if score > self._best_trial_score:
                self._best_trial_score = score
                self._best_trial_key = combo_key

            # Record trial
            trial_num = len(self._trial_history) + 1
            self._trial_history.append(
                {
                    "trial": trial_num,
                    "instr_idx": instr_idx,
                    "demo_idx": demo_idx,
                    "score": score,
                }
            )

            # Progress update
            demo_info = (
                f"{len(demo_set.demos)} demos" if demo_set.demos else "0-shot"
            )
            self._update_status(
                f"Trial {trial_num}/{self.num_trials} - "
                f"Instr {instr_idx}, {demo_info} - Score: {score:.4f}",
                trial_num,
            )

            # Periodic full evaluation
            if trial_num % self.minibatch_full_eval_steps == 0:
                best_instr, best_demo = self._best_trial_key
                if (best_instr, best_demo) not in self._full_eval_cache:
                    best_config = self._get_config_by_index(best_instr)
                    best_demo_set = self._demo_sets[best_demo]
                    self._full_evaluate(best_config, best_demo_set, goldens)

            return score

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.num_trials,
            show_progress_bar=False,
        )

        # Return the best combination
        return (
            study.best_params["instr_idx"],
            study.best_params["demo_idx"],
        )

    async def _a_run_bayesian_optimization(
        self,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[int, int]:
        """
        Async version of Bayesian Optimization.
        """
        num_instructions = len(self._instruction_candidates)
        num_demo_sets = len(self._demo_sets)

        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        for trial_num in range(1, self.num_trials + 1):
            trial = study.ask()

            # Sample indices
            instr_idx = trial.suggest_int("instr_idx", 0, num_instructions - 1)
            demo_idx = trial.suggest_int("demo_idx", 0, num_demo_sets - 1)

            # Get config and demos
            config = self._get_config_by_index(instr_idx)
            demo_set = self._demo_sets[demo_idx]
            rendered_config = self._render_config_with_demos(config, demo_set)

            # Score on minibatch
            minibatch = self._draw_minibatch(goldens)
            score = await self.scorer.a_score_minibatch(
                rendered_config, minibatch
            )

            # Track scores
            combo_key = (instr_idx, demo_idx)
            if combo_key not in self._combination_scores:
                self._combination_scores[combo_key] = []
            self._combination_scores[combo_key].append(score)

            # Update best
            if score > self._best_trial_score:
                self._best_trial_score = score
                self._best_trial_key = combo_key

            # Record trial
            self._trial_history.append(
                {
                    "trial": trial_num,
                    "instr_idx": instr_idx,
                    "demo_idx": demo_idx,
                    "score": score,
                }
            )

            # Tell Optuna the result
            study.tell(trial, score)

            # Progress update
            demo_info = (
                f"{len(demo_set.demos)} demos" if demo_set.demos else "0-shot"
            )
            self._update_status(
                f"Trial {trial_num}/{self.num_trials} - "
                f"Instr {instr_idx}, {demo_info} - Score: {score:.4f}",
                trial_num,
            )

            # Periodic full evaluation
            if trial_num % self.minibatch_full_eval_steps == 0:
                best_instr, best_demo = self._best_trial_key
                if (best_instr, best_demo) not in self._full_eval_cache:
                    best_config = self._get_config_by_index(best_instr)
                    best_demo_set = self._demo_sets[best_demo]
                    await self._a_full_evaluate(
                        best_config, best_demo_set, goldens
                    )

        return (
            study.best_params["instr_idx"],
            study.best_params["demo_idx"],
        )

    ############################
    # Full Evaluation          #
    ############################

    def _full_evaluate(
        self,
        config: PromptConfiguration,
        demo_set: DemoSet,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> None:
        """Perform full evaluation on all goldens."""
        # Find the indices for this combination
        instr_idx = self._instruction_candidates.index(config)
        demo_idx = self._demo_sets.index(demo_set)
        combo_key = (instr_idx, demo_idx)

        if combo_key in self._full_eval_cache:
            return

        # Render with demos
        rendered_config = self._render_config_with_demos(config, demo_set)

        # Register the rendered config
        self.prompt_configurations_by_id[rendered_config.id] = rendered_config
        self.parents_by_id[rendered_config.id] = config.id

        # Score on full set
        scores = self.scorer.score_pareto(rendered_config, goldens)
        self.pareto_score_table[rendered_config.id] = scores

        # Cache the result
        self._full_eval_cache[combo_key] = rendered_config.id

    async def _a_full_evaluate(
        self,
        config: PromptConfiguration,
        demo_set: DemoSet,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> None:
        """Async full evaluation."""
        instr_idx = self._instruction_candidates.index(config)
        demo_idx = self._demo_sets.index(demo_set)
        combo_key = (instr_idx, demo_idx)

        if combo_key in self._full_eval_cache:
            return

        rendered_config = self._render_config_with_demos(config, demo_set)
        self.prompt_configurations_by_id[rendered_config.id] = rendered_config
        self.parents_by_id[rendered_config.id] = config.id

        scores = await self.scorer.a_score_pareto(rendered_config, goldens)
        self.pareto_score_table[rendered_config.id] = scores
        self._full_eval_cache[combo_key] = rendered_config.id

    ############################
    # Result Building          #
    ############################

    def _best_by_aggregate(self) -> PromptConfiguration:
        """Return the best candidate based on full evaluation scores."""
        if not self.pareto_score_table:
            # Fall back to best by trial scores
            best_instr, best_demo = self._best_trial_key
            config = self._get_config_by_index(best_instr)
            demo_set = self._demo_sets[best_demo]
            return self._render_config_with_demos(config, demo_set)

        best_id: Optional[PromptConfigurationId] = None
        best_score = float("-inf")

        for config_id, scores in self.pareto_score_table.items():
            agg_score = self.aggregate_instances(scores)
            if agg_score > best_score:
                best_score = agg_score
                best_id = config_id

        if best_id is None:
            best_instr, best_demo = self._best_trial_key
            config = self._get_config_by_index(best_instr)
            demo_set = self._demo_sets[best_demo]
            return self._render_config_with_demos(config, demo_set)

        return self.prompt_configurations_by_id[best_id]

    def _build_result(
        self,
        best: PromptConfiguration,
    ) -> Tuple[Prompt, OptimizationReport]:
        """Build the optimization result."""
        prompt_config_snapshots = build_prompt_config_snapshots(
            self.prompt_configurations_by_id
        )

        report = OptimizationReport(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_iterations=self._trial_history,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
            prompt_configurations=prompt_config_snapshots,
        )

        return best.prompts[self.SINGLE_MODULE_ID], report

    ############################
    # Status Updates           #
    ############################

    def _update_status(self, message: str, step: int) -> None:
        """Send status update via callback."""
        if self.status_callback is not None:
            self.status_callback(
                RunnerStatusType.PROGRESS,
                step_index=step,
                total_steps=self.num_trials,
                detail=message,
            )
