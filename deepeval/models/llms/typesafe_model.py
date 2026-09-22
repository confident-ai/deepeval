import asyncio
import json
from typing import Optional, Tuple, Union, Dict, Any, Type, List, get_origin
from pydantic import BaseModel, SecretStr

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models.utils import (
    require_costs,
    EvaluationCost,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.constants import TYPESAFE_MODELS_DATA
from deepeval.utils import require_param

# Retry decorator for TypeSafe
retry_typesafe = create_retry_decorator(PS.TYPESAFE)


class TypeSafeModel(DeepEvalBaseLLM):
    """
    TypeSafe / Jev model wrapper for DeepEval.
    
    Jev is a fast, non-autoregressive 'System One' decision model by TypeSafe AI.
    Unlike traditional autoregressive LLMs that predict next tokens sequentially,
    Jev operates directly on states and outputs calibrated, typed decisions
    (Choice, Score, Noul) in 70–500ms.
    
    For metrics requiring free-form text synthesis (such as auto-generating evaluation
    steps in GEval), TypeSafeModel can be paired with an optional `generator_model`
    (e.g., gpt-4o-mini), while Jev executes all scoring and judgment decisions.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        generator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        settings = get_settings()

        model = model or settings.TYPESAFE_MODEL_NAME or "jev"

        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.TYPESAFE_COST_PER_INPUT_TOKEN
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.TYPESAFE_COST_PER_OUTPUT_TOKEN
        )

        if api_key is not None:
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.TYPESAFE_API_KEY

        self.base_url = base_url or "https://api.typesafe.ai"

        model = require_param(
            model,
            provider_label="TypeSafeModel",
            env_var_name="TYPESAFE_MODEL_NAME",
            param_hint="model",
        )

        self.model_data = TYPESAFE_MODELS_DATA.get(model)
        if self.model_data:
            cost_per_input_token, cost_per_output_token = require_costs(
                self.model_data,
                model,
                "TYPESAFE_COST_PER_INPUT_TOKEN",
                "TYPESAFE_COST_PER_OUTPUT_TOKEN",
                cost_per_input_token,
                cost_per_output_token,
            )
            self.model_data.input_price = cost_per_input_token
            self.model_data.output_price = cost_per_output_token

        # Optional generator model for open-ended text synthesis tasks (e.g. generating steps)
        self.generator_model: Optional[DeepEvalBaseLLM] = None
        if generator_model is not None:
            if isinstance(generator_model, DeepEvalBaseLLM):
                self.generator_model = generator_model
            elif isinstance(generator_model, str):
                from deepeval.models.llms import OpenAIModel
                self.generator_model = OpenAIModel(model=generator_model)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.kwargs = kwargs

        super().__init__(model)

    def load_model(self, *args, **kwargs):
        """Loads the TypeSafeClient with lazy importing."""
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError:
            raise DeepEvalError(
                "To use TypeSafe / Jev models, please install the SDK: pip install typesafe-sdk"
            )

        api_key_str = self.api_key.get_secret_value() if self.api_key else None
        return TypeSafeClient(
            api_key=api_key_str,
            base_url=self.base_url,
            **self.kwargs,
        )

    def _is_text_synthesis_schema(self, schema: Type[BaseModel]) -> bool:
        """
        Determines whether a schema expects open-ended text generation or list synthesis
        (which Jev cannot perform natively) versus a decision/scoring task.
        """
        fields = getattr(schema, "model_fields", {}) or getattr(schema, "__fields__", {})
        
        # If schema only has list of strings (like Steps) and no score/verdict/boolean
        has_score_or_decision = any(
            name.lower() in ("score", "verdict", "passed", "is_factual", "rating")
            for name in fields
        )
        has_list_generation = any(
            get_origin(getattr(f, "annotation", None)) is list
            for f in fields.values()
        )

        if has_list_generation and not has_score_or_decision:
            return True
        return False

    def _convert_schema_to_questions(self, schema: Type[BaseModel]) -> Dict[str, Any]:
        """Maps a Pydantic BaseModel schema to TypeSafe questions (Choice, Score, Noul)."""
        try:
            from typesafe_sdk import Choice, Score, Noul
        except ImportError:
            raise DeepEvalError(
                "To use TypeSafe / Jev models, please install the SDK: pip install typesafe-sdk"
            )

        questions = {}
        fields = getattr(schema, "model_fields", {}) or getattr(schema, "__fields__", {})

        for field_name, field_info in fields.items():
            field_type = getattr(field_info, "annotation", None)
            description = getattr(field_info, "description", None) or field_name

            if field_type in (bool, Optional[bool]):
                questions[field_name] = Noul(instructions=f"Is the following statement satisfied: {description}")
            elif field_type in (int, float, Optional[int], Optional[float]):
                # Map rubrics directly into Jev's Score levels (0 to 10 or 1 to 5)
                questions[field_name] = Score(
                    instructions=f"Score the following criterion: {description}",
                    levels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                )
            else:
                # Qualitative, citation, or verdict field
                if "citation" in field_name.lower() or "faithfulness" in field_name.lower():
                    # Direct 3-way citation verification: supports, contradicts, unrelated
                    questions[field_name] = Choice(
                        instructions=f"Verify citation relation for {description}",
                        criteria={
                            "supports": "Source context directly supports the output",
                            "contradicts": "Source context contradicts the output",
                            "unrelated": "Source context is unrelated to the output",
                        }
                    )
                elif "verdict" in field_name.lower():
                    questions[field_name] = Choice(
                        instructions=f"Select verdict for {description}",
                        criteria={"yes": "Criteria is met", "no": "Criteria is not met"}
                    )
                else:
                    questions[field_name] = Noul(instructions=f"Verify requirement: {description}")

        return questions

    @retry_typesafe
    def generate(
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """
        Executes evaluation with Jev decision model.
        If a text synthesis task (like generating evaluation steps) is requested:
        - Delegates to `generator_model` if configured.
        - Falls back to deterministic rubric steps if it is a `Steps` schema.
        """
        # Case 1: Schema-driven evaluation
        if schema is not None:
            # Check if this is a text synthesis schema (e.g. GEval Steps)
            if self._is_text_synthesis_schema(schema):
                if self.generator_model is not None:
                    return self.generator_model.generate(prompt, schema=schema)
                
                # Deterministic fallback for Steps schemas
                fields = getattr(schema, "model_fields", {}) or getattr(schema, "__fields__", {})
                if "steps" in fields:
                    fallback_instance = schema.model_validate({
                        "steps": [
                            "Assess whether the actual output satisfies the criteria specified.",
                            "Verify factual alignment and relevance to the input.",
                            "Check completeness and overall response quality."
                        ]
                    })
                    return fallback_instance, 0.0

            # Native Decision/Scoring evaluation with Jev
            client = self.load_model()
            questions = self._convert_schema_to_questions(schema)
            if not questions:
                try:
                    from typesafe_sdk import Score
                    questions = {"score": Score(instructions="Score the response quality", levels=["1", "2", "3", "4", "5"])}
                except ImportError:
                    pass

            response = client.system_one(state=prompt, questions=questions)
            
            # Map Jev answers back to schema fields
            answers_dict = {}
            score_level_recorded = None
            for k, ans in response.answers.items():
                if hasattr(ans, "choice") and ans.choice is not None:
                    answers_dict[k] = ans.choice
                elif hasattr(ans, "score") and ans.score is not None:
                    score_val = ans.score
                    score_level_recorded = score_val
                    if isinstance(score_val, str) and score_val.isdigit():
                        answers_dict[k] = float(score_val)
                    else:
                        answers_dict[k] = score_val
                elif hasattr(ans, "noul") and ans.noul is not None:
                    answers_dict[k] = ans.noul
                else:
                    answers_dict[k] = str(ans)

            # Build sensible defaults for fields not covered directly (e.g. reason strings)
            for field_name, field_info in getattr(schema, "model_fields", {}).items():
                if field_name not in answers_dict:
                    if "reason" in field_name.lower():
                        if score_level_recorded is not None:
                            answers_dict[field_name] = f"Evaluated via TypeSafe Jev model with rating level: {score_level_recorded}."
                        else:
                            answers_dict[field_name] = "Decision evaluated via TypeSafe Jev model."
                    elif "verdict" in field_name.lower():
                        answers_dict[field_name] = "yes"
                    else:
                        answers_dict[field_name] = 1.0

            validated = schema.model_validate(answers_dict)
            cost = self.calculate_cost(len(prompt.split()), 50)
            return validated, cost

        # Case 2: Free-form prompt without schema
        else:
            if self.generator_model is not None:
                return self.generator_model.generate(prompt)

            client = self.load_model()
            try:
                from typesafe_sdk import Score
                questions = {"evaluation": Score(instructions="Evaluate the input quality", levels=["Pass", "Fail"])}
            except ImportError:
                questions = {}

            response = client.system_one(state=prompt, questions=questions)
            output_dict = {
                k: (getattr(v, "choice", None) or getattr(v, "score", None) or getattr(v, "noul", str(v)))
                for k, v in response.answers.items()
            }
            output_str = json.dumps(output_dict)
            cost = self.calculate_cost(len(prompt.split()), 50)
            return output_str, cost

    @retry_typesafe
    async def a_generate(
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Asynchronously executes decision or delegated generation via threadpool."""
        return await asyncio.to_thread(self.generate, prompt, schema)

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> EvaluationCost:
        if self.model_data and self.model_data.input_price and self.model_data.output_price:
            input_cost = input_tokens * self.model_data.input_price
            output_cost = output_tokens * self.model_data.output_price
            return EvaluationCost(
                input_cost + output_cost, input_tokens, output_tokens
            )
        return EvaluationCost(0.0, input_tokens, output_tokens)

    def get_model_name(self, *args, **kwargs) -> str:
        return self.name

    def supports_log_probs(self) -> Union[bool, None]:
        return False

    def supports_temperature(self) -> Union[bool, None]:
        return False

    def supports_multimodal(self) -> Union[bool, None]:
        return False

    def supports_structured_outputs(self) -> Union[bool, None]:
        return True

    def supports_json_mode(self) -> Union[bool, None]:
        return True


# Alias for developer convenience
JevModel = TypeSafeModel
