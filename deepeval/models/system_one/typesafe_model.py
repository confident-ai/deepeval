from typing import Any, Dict, Optional, Tuple

from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseSystemOneModel
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.system_one.constants import (
    DEFAULT_TYPESAFE_MODEL,
    TYPESAFE_MODELS_DATA,
)
from deepeval.models.system_one.limits import check_context_budget
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneAnswers,
    SystemOneQuestion,
)
from deepeval.models.utils import EvaluationCost, require_secret_api_key
from deepeval.utils import require_dependency

retry_typesafe = create_retry_decorator(PS.TYPESAFE)


class TypeSafeModel(DeepEvalBaseSystemOneModel):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        **kwargs,
    ):
        settings = get_settings()

        if api_key is not None:
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.TYPESAFE_API_KEY

        model = model or settings.TYPESAFE_MODEL_NAME or DEFAULT_TYPESAFE_MODEL

        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.TYPESAFE_COST_PER_INPUT_TOKEN
        )
        if cost_per_input_token is not None and cost_per_input_token < 0:
            raise DeepEvalError("TYPESAFE_COST_PER_INPUT_TOKEN must be >= 0.")

        self.model_data = TYPESAFE_MODELS_DATA.get(model)
        if cost_per_input_token is not None:
            self.model_data.input_price = cost_per_input_token
        self.model_data.output_price = 0.0

        self.kwargs = kwargs
        self._async_client = None
        super().__init__(model)

    ###############################################
    # Decide
    ###############################################

    @retry_typesafe
    def decide(
        self,
        state: Any,
        questions: Dict[str, SystemOneQuestion],
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        sdk_questions = self._to_sdk_questions(questions)
        check_context_budget(state, sdk_questions)
        client = self.load_model()
        response = client.system_one(state, sdk_questions, model=self.name)
        return self._from_sdk_response(response)

    @retry_typesafe
    async def a_decide(
        self,
        state: Any,
        questions: Dict[str, SystemOneQuestion],
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        sdk_questions = self._to_sdk_questions(questions)
        check_context_budget(state, sdk_questions)
        client = self.load_model(async_mode=True)
        response = await client.system_one(
            state, sdk_questions, model=self.name
        )
        return self._from_sdk_response(response)

    ###############################################
    # Translation
    ###############################################

    def _to_sdk_questions(
        self, questions: Dict[str, SystemOneQuestion]
    ) -> Dict[str, Any]:
        if not questions:
            raise DeepEvalError(
                "TypeSafeModel.decide requires at least one question."
            )
        sdk: Dict[str, Any] = {}
        for key, question in questions.items():
            if isinstance(question, NoulQuestion):
                criteria = None
                if question.true is not None or question.false is not None:
                    criteria = {
                        "true": question.true,
                        "false": question.false,
                    }
                sdk[key] = {
                    "type": "noul",
                    "instructions": question.instructions,
                    "criteria": criteria,
                }
            elif isinstance(question, ChoiceQuestion):
                sdk[key] = {
                    "type": "choice",
                    "instructions": question.instructions,
                    "criteria": dict(question.options),
                }
            elif isinstance(question, ScoreQuestion):
                sdk[key] = {
                    "type": "score",
                    "instructions": question.instructions,
                    "criteria": list(question.levels),
                }
            else:
                raise DeepEvalError(
                    f"Unsupported System One question type: {type(question)}"
                )
            if sdk[key]["criteria"] is None:
                del sdk[key]["criteria"]
        return sdk

    def _from_sdk_response(
        self, response: Any
    ) -> Tuple[SystemOneAnswers, Optional[float]]:
        answers = SystemOneAnswers()
        for key, answer in response.nouls.items():
            answers.nouls[key] = NoulAnswer(probability=answer.noul)
        for key, answer in response.choices.items():
            answers.choices[key] = ChoiceAnswer(
                choice=answer.choice,
                probabilities=dict(answer.probabilities),
                confidence=answer.confidence,
            )
        for key, answer in response.scores.items():
            answers.scores[key] = ScoreAnswer(
                score=answer.score,
                probabilities={
                    int(level): p for level, p in answer.probabilities.items()
                },
                confidence=answer.confidence,
            )
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) or 0
        output_tokens = getattr(usage, "output_tokens", None) or 0
        return answers, self.calculate_cost(input_tokens, output_tokens)

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(
        self, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        if self.model_data.input_price is None:
            return None
        return EvaluationCost(
            input_tokens * self.model_data.input_price,
            input_tokens,
            output_tokens,
        )

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        module = require_dependency(
            "typesafe_sdk",
            provider_label="TypeSafeModel",
            install_hint="Install it with `pip install typesafe-sdk`.",
        )
        if not async_mode:
            return self._build_client(module, module.TypeSafeClient)
        if self._async_client is None:
            self._async_client = self._build_client(
                module, module.AsyncTypeSafeClient
            )
        return self._async_client

    def _build_client(self, module, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="TypeSafe AI",
            env_var_name="TYPESAFE_API_KEY",
            param_hint="`api_key` to TypeSafeModel(...)",
        )
        kw = dict(api_key=api_key, model=self.name, **self.kwargs)
        if not sdk_retries_for(PS.TYPESAFE) and "retry" not in kw:
            kw["retry"] = module.RetryPolicy(max_retries=0)
        return cls(**kw)

    def get_model_name(self):
        return f"{self.name} (TypeSafe AI)"
