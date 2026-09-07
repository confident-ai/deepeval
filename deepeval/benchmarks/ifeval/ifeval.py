import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from deepeval.benchmarks.base_benchmark import (DeepEvalBaseBenchmark,
                                                DeepEvalBaseBenchmarkResult)
from deepeval.benchmarks.schema import StringSchema
from deepeval.dataset import Golden
from deepeval.models import DeepEvalBaseLLM
from deepeval.telemetry import capture_benchmark_run
from deepeval.utils import make_model_config


class IFEvalResult(DeepEvalBaseBenchmarkResult):
    model_config = make_model_config(arbitrary_types_allowed=True)
    instruction_breakdown: dict[str, Any]
    predictions: "pd.DataFrame"


# class IFEvalInstructionVerifier:
#     """
#     Verifies instruction compliance for IFEval benchmark.

#     Implements rule-based verification for various instruction types including
#     punctuation constraints, length constraints, format requirements, and content rules.
#     """

#     @staticmethod
#     def verify_punctuation_constraints(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> bool:
#         """Verify punctuation-related constraints."""
#         if instruction_id == "punctuation:no_comma":
#             return "," not in response
#         elif instruction_id == "punctuation:no_period":
#             return "." not in response
#         elif instruction_id == "punctuation:no_question_mark":
#             return "?" not in response
#         elif instruction_id == "punctuation:no_exclamation_mark":
#             return "!" not in response
#         return True

#     @staticmethod
#     def verify_length_constraints(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> bool:
#         """Verify length-related constraints."""
#         if instruction_id == "length_constraints:number_words":
#             num_words = kwargs.get("num_words")
#             relation = kwargs.get("relation", "exactly")

#             if num_words is None:
#                 return True

#             word_count = len(response.split())

#             if relation == "exactly":
#                 return word_count == num_words
#             elif relation == "at least":
#                 return word_count >= num_words
#             elif relation == "less than":
#                 return word_count < num_words
#             elif relation == "more than":
#                 return word_count > num_words

#         elif instruction_id == "length_constraints:number_characters":
#             num_chars = kwargs.get("num_chars")
#             relation = kwargs.get("relation", "exactly")

#             if num_chars is None:
#                 return True

#             char_count = len(response)

#             if relation == "exactly":
#                 return char_count == num_chars
#             elif relation == "at least":
#                 return char_count >= num_chars
#             elif relation == "less than":
#                 return char_count < num_chars
#             elif relation == "more than":
#                 return char_count > num_chars

#         elif instruction_id == "length_constraints:number_sentences":
#             num_sentences = kwargs.get("num_sentences")
#             relation = kwargs.get("relation", "exactly")

#             if num_sentences is None:
#                 return True

#             sentences = re.split(r"[.!?]+", response)
#             sentence_count = len([s for s in sentences if s.strip()])

#             if relation == "exactly":
#                 return sentence_count == num_sentences
#             elif relation == "at least":
#                 return sentence_count >= num_sentences
#             elif relation == "less than":
#                 return sentence_count < num_sentences
#             elif relation == "more than":
#                 return sentence_count > num_sentences

#         return True

#     @staticmethod
#     def verify_format_constraints(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> bool:
#         """Verify format-related constraints."""
#         if instruction_id == "detectable_format:json":
#             try:
#                 json.loads(response)
#                 return True
#             except (json.JSONDecodeError, ValueError):
#                 return False

#         elif instruction_id == "detectable_format:list":
#             lines = response.strip().split("\n")
#             return (
#                 len(lines) > 1
#                 or response.strip().startswith("-")
#                 or response.strip().startswith("*")
#                 or response.strip().startswith("1.")
#                 or response.strip().startswith("•")
#             )

#         elif instruction_id == "detectable_format:number_bullets":
#             num_bullets = kwargs.get("num_bullets")
#             if num_bullets is None:
#                 return True

#             bullet_patterns = [r"^\s*[-*•]\s+", r"^\s*\d+\.\s+"]
#             bullet_count = 0

#             for line in response.split("\n"):
#                 for pattern in bullet_patterns:
#                     if re.match(pattern, line):
#                         bullet_count += 1
#                         break

#             return bullet_count >= num_bullets

#         elif instruction_id == "detectable_format:number_highlighted_sections":
#             num_highlights = kwargs.get("num_highlights")
#             if num_highlights is None:
#                 return True

#             highlight_pattern = r"\*[^*]+\*"
#             highlights = re.findall(highlight_pattern, response)
#             return len(highlights) >= num_highlights

#         elif instruction_id == "detectable_format:title":
#             title_pattern = r"<<[^>]+>>"
#             return bool(re.search(title_pattern, response))

#         return True

#     @staticmethod
#     def verify_case_constraints(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> bool:
#         """Verify case-related constraints."""
#         if instruction_id == "change_case:english_lowercase":
#             return response.lower() == response and response.islower()

#         elif instruction_id == "change_case:english_uppercase":
#             return response.upper() == response and response.isupper()

#         elif instruction_id == "change_case:english_titlecase":
#             return response.istitle()

#         return True

#     @staticmethod
#     def verify_startend_constraints(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> bool:
#         """Verify start/end constraints."""
#         if instruction_id == "startend:start_with":
#             start_text = kwargs.get("start_text")
#             if start_text is None:
#                 return True
#             return response.strip().startswith(start_text)

#         elif instruction_id == "startend:end_with":
#             end_text = kwargs.get("end_text")
#             if end_text is None:
#                 return True
#             return response.strip().endswith(end_text)

#         return True

#     @staticmethod
#     def verify_keywords_constraints(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> bool:
#         """Verify keyword constraints."""
#         if instruction_id == "keywords:must_include":
#             required_keywords = kwargs.get("keywords", [])
#             response_lower = response.lower()
#             return all(
#                 keyword.lower() in response_lower
#                 for keyword in required_keywords
#             )

#         elif instruction_id == "keywords:must_not_include":
#             forbidden_keywords = kwargs.get("keywords", [])
#             response_lower = response.lower()
#             return not any(
#                 keyword.lower() in response_lower
#                 for keyword in forbidden_keywords
#             )

#         return True

#     @staticmethod
#     def verify_content_constraints(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> bool:
#         """Verify content-related constraints."""
#         if instruction_id == "detectable_content:keyword_frequency":
#             keyword = kwargs.get("keyword")
#             frequency = kwargs.get("frequency")
#             relation = kwargs.get("relation", "exactly")

#             if keyword is None or frequency is None:
#                 return True

#             keyword_count = response.lower().count(keyword.lower())

#             if relation == "exactly":
#                 return keyword_count == frequency
#             elif relation == "at least":
#                 return keyword_count >= frequency
#             elif relation == "less than":
#                 return keyword_count < frequency
#             elif relation == "more than":
#                 return keyword_count > frequency

#         elif instruction_id == "detectable_content:forbidden_words":
#             forbidden_words = kwargs.get("forbidden_words", [])
#             for word in forbidden_words:
#                 if word.lower() in response.lower():
#                     return False
#             return True

#         elif instruction_id == "detectable_content:number_placeholders":
#             num_placeholders = kwargs.get("num_placeholders")
#             if num_placeholders is None:
#                 return True

#             placeholder_pattern = r"\[[^\]]+\]"
#             placeholders = re.findall(placeholder_pattern, response)
#             return len(placeholders) >= num_placeholders

#         elif instruction_id == "detectable_content:postscript":
#             postscript_marker = kwargs.get("postscript_marker", "P.S.")
#             return postscript_marker in response

#         elif instruction_id == "detectable_content:first_word":
#             first_word = kwargs.get("first_word")
#             if first_word is None:
#                 return True

#             response_words = response.strip().split()
#             return (
#                 response_words
#                 and response_words[0].lower() == first_word.lower()
#             )

#         return True

#     @staticmethod
#     def verify_structural_constraints(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> bool:
#         """Verify structural constraints."""
#         if instruction_id == "structural_constraints:number_paragraphs":
#             num_paragraphs = kwargs.get("num_paragraphs")
#             relation = kwargs.get("relation", "exactly")

#             if num_paragraphs is None:
#                 return True

#             paragraphs = [p for p in response.split("\n\n") if p.strip()]
#             paragraph_count = len(paragraphs)

#             if relation == "exactly":
#                 return paragraph_count == num_paragraphs
#             elif relation == "at least":
#                 return paragraph_count >= num_paragraphs
#             elif relation == "less than":
#                 return paragraph_count < num_paragraphs
#             elif relation == "more than":
#                 return paragraph_count > num_paragraphs

#         elif instruction_id == "structural_constraints:number_sections":
#             num_sections = kwargs.get("num_sections")
#             section_spliter = kwargs.get("section_spliter", "---")

#             if num_sections is None:
#                 return True

#             sections = [s for s in response.split(section_spliter) if s.strip()]
#             section_count = len(sections)
#             return section_count >= num_sections

#         return True

#     @staticmethod
#     def verify_combination_constraints(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> bool:
#         """Verify combination constraints."""
#         if instruction_id == "combination:repeat_prompt":
#             prompt_to_repeat = kwargs.get("prompt_to_repeat")
#             if prompt_to_repeat is None:
#                 return True

#             return prompt_to_repeat in response

#         return True

#     @staticmethod
#     def verify_instruction_compliance(
#         response: str, instruction_id: str, kwargs: Dict[str, Any]
#     ) -> Tuple[bool, str]:
#         """Verify compliance with a single instruction."""
#         try:
#             if instruction_id.startswith("punctuation:"):
#                 result = (
#                     IFEvalInstructionVerifier.verify_punctuation_constraints(
#                         response, instruction_id, kwargs
#                     )
#                 )
#             elif instruction_id.startswith("length_constraints:"):
#                 result = IFEvalInstructionVerifier.verify_length_constraints(
#                     response, instruction_id, kwargs
#                 )
#             elif instruction_id.startswith("detectable_format:"):
#                 result = IFEvalInstructionVerifier.verify_format_constraints(
#                     response, instruction_id, kwargs
#                 )
#             elif instruction_id.startswith("detectable_content:"):
#                 result = IFEvalInstructionVerifier.verify_content_constraints(
#                     response, instruction_id, kwargs
#                 )
#             elif instruction_id.startswith("structural_constraints:"):
#                 result = (
#                     IFEvalInstructionVerifier.verify_structural_constraints(
#                         response, instruction_id, kwargs
#                     )
#                 )
#             elif instruction_id.startswith("combination:"):
#                 result = (
#                     IFEvalInstructionVerifier.verify_combination_constraints(
#                         response, instruction_id, kwargs
#                     )
#                 )
#             elif instruction_id.startswith("change_case:"):
#                 result = IFEvalInstructionVerifier.verify_case_constraints(
#                     response, instruction_id, kwargs
#                 )
#             elif instruction_id.startswith("startend:"):
#                 result = IFEvalInstructionVerifier.verify_startend_constraints(
#                     response, instruction_id, kwargs
#                 )
#             elif instruction_id.startswith("keywords:"):
#                 result = IFEvalInstructionVerifier.verify_keywords_constraints(
#                     response, instruction_id, kwargs
#                 )
#             else:
#                 return False, f"Unknown instruction type: {instruction_id}"

#             reason = f"Instruction '{instruction_id}' {'PASSED' if result else 'FAILED'}"
#             return result, reason

#         except Exception as e:
#             return (
#                 False,
#                 f"Error verifying instruction '{instruction_id}': {str(e)}",
#             )


class IFEvalInstructionVerifier:
    """
    Verifier for Google IFEval instruction compliance.
    Directly implements the 29 canonical instruction IDs without silent pass fallthroughs.
    """

    # --- 1. Keywords Constraints ---
    @staticmethod
    def verify_keywords_existence(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        for kw in kwargs.get("keywords", []):
            if not re.search(rf"\b{re.escape(kw)}\b", response, re.IGNORECASE):
                return False, f"Missing required keyword: '{kw}'"
        return True, "Instruction 'keywords:existence' PASSED"

    @staticmethod
    def verify_keywords_frequency(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        kw, freq, rel = (
            kwargs.get("keyword", ""),
            kwargs.get("frequency", 0),
            kwargs.get("relation", "gte"),
        )
        count = len(
            re.findall(rf"\b{re.escape(kw)}\b", response, re.IGNORECASE)
        )
        if (
            (rel == "gte" and count >= freq)
            or (rel == "lte" and count <= freq)
            or (rel == "eq" and count == freq)
        ):
            return True, f"Instruction 'keywords:frequency' PASSED"
        return False, f"Keyword '{kw}' count {count}, expected {rel} {freq}"

    @staticmethod
    def verify_forbidden_words(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        for word in kwargs.get("forbidden_words", []):
            if re.search(rf"\b{re.escape(word)}\b", response, re.IGNORECASE):
                return False, f"Forbidden word found: '{word}'"
        return True, "Instruction 'keywords:forbidden_words' PASSED"

    @staticmethod
    def verify_letter_frequency(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        let, freq, rel = (
            kwargs.get("letter", ""),
            kwargs.get("let_frequency", 0),
            kwargs.get("let_relation", "gte"),
        )
        count = response.lower().count(let.lower())
        if (
            (rel in ("at least", "gte") and count >= freq)
            or (rel in ("less than", "lte") and count < freq)
            or (rel in ("equals", "eq") and count == freq)
        ):
            return True, f"Instruction 'keywords:letter_frequency' PASSED"
        return False, f"Letter '{let}' count {count}, expected {rel} {freq}"

    @staticmethod
    def verify_key_sentences(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        sentences = kwargs.get("key_sentences", [])
        num_req = kwargs.get("num_sentences", len(sentences))
        count = sum(1 for sent in sentences if sent.strip() in response)
        if count >= num_req:
            return True, f"Instruction 'keywords:key_sentences' PASSED"
        return False, f"Matched {count}/{num_req} key sentences"

    # --- 2. Format Constraints ---
    @staticmethod
    def verify_number_bullet_lists(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        req = kwargs.get("num_bullet_lists", 0)
        count = len(re.findall(r"(?m)^\s*(?:[\*\-\•]|\d+\.)\s+", response))
        return (
            (True, "PASSED")
            if count >= req
            else (False, f"Found {count} bullets, expected >= {req}")
        )

    @staticmethod
    def verify_json_format(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        clean_text = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```", response.strip()
        )
        clean_text = (
            clean_text.group(1).strip() if clean_text else response.strip()
        )
        try:
            json.loads(clean_text)
            return True, "Instruction 'detectable_format:json_format' PASSED"
        except ValueError as e:
            return False, f"Invalid JSON: {str(e)}"

    @staticmethod
    def verify_multiple_sections(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        req, sep = kwargs.get("num_sections", 0), kwargs.get(
            "section_spliter", "Section"
        )
        count = len(
            re.findall(
                rf"(?i)(?:^|\n)\s*(?:#+\s*)?{re.escape(sep)}\s+[\w\d]+",
                response,
            )
        )
        return (
            (True, "PASSED")
            if count >= req
            else (False, f"Found {count} sections, expected {req}")
        )

    @staticmethod
    def verify_constrained_response(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        candidates = [c.strip().lower() for c in kwargs.get("candidates", [])]
        return (
            (True, "PASSED")
            if response.strip().lower() in candidates
            else (False, "Response not in allowed candidates")
        )

    # --- 3. Case & Length Constraints ---
    @staticmethod
    def verify_english_capital(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        letters = re.findall(r"[a-zA-Z]", response)
        return (
            (True, "PASSED")
            if letters and all(c.isupper() for c in letters)
            else (False, "Contains non-capital characters")
        )

    @staticmethod
    def verify_english_lowercase(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        letters = re.findall(r"[a-zA-Z]", response)
        return (
            (True, "PASSED")
            if letters and all(c.islower() for c in letters)
            else (False, "Contains uppercase characters")
        )

    @staticmethod
    def verify_number_paragraphs(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        req = kwargs.get("num_paragraphs", 0)
        count = len(
            [p for p in re.split(r"\n\s*\n", response.strip()) if p.strip()]
        )
        return (
            (True, "PASSED")
            if count == req
            else (False, f"Found {count} paragraphs, expected {req}")
        )

    @staticmethod
    def verify_nth_paragraph_first_word(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        nth, word = (
            kwargs.get("nth_paragraph", 1),
            kwargs.get("first_word", "").strip().lower(),
        )
        paras = [
            p.strip()
            for p in re.split(r"\n\s*\n", response.strip())
            if p.strip()
        ]
        if len(paras) < nth:
            return False, f"Paragraph {nth} missing"
        first_match = re.findall(r"\b\w+\b", paras[nth - 1])
        return (
            (True, "PASSED")
            if first_match and first_match[0].lower() == word
            else (False, "First word mismatch")
        )

    # --- 4. Start/End & Quoting ---
    @staticmethod
    def verify_end_checker(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        phrase = kwargs.get("end_phrase", "").strip()
        return (
            (True, "PASSED")
            if response.strip().endswith(phrase)
            else (False, "End phrase mismatch")
        )

    @staticmethod
    def verify_quotation(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        c = response.strip()
        return (
            (True, "PASSED")
            if (c.startswith('"') and c.endswith('"'))
            or (c.startswith("'") and c.endswith("'"))
            else (False, "Not wrapped in quotes")
        )

    # --- 5. Language ---
    @staticmethod
    def verify_response_language(
        response: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        target = kwargs.get("language", "en").lower()
        try:
            from langdetect import detect

            return (
                (True, "PASSED")
                if detect(response) == target
                else (False, "Language mismatch")
            )
        except ImportError:
            return True, "Passed (langdetect missing)"

    @classmethod
    def verify_instruction_compliance(
        cls, response: str, instruction_id: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        # Lookup happens against the module-level dispatch map so handler
        # objects are plain callables (not staticmethod descriptors).
        handler = IFEVAL_DISPATCH_MAP.get(instruction_id)

        if handler is None:
            return (
                False,
                f"Unknown or unsupported instruction ID: '{instruction_id}'",
            )

        return handler(response, kwargs)


# Module-level dispatch map with plain callables to avoid staticmethod descriptor issues
IFEVAL_DISPATCH_MAP = {
    "keywords:existence": IFEvalInstructionVerifier.verify_keywords_existence,
    "keywords:frequency": IFEvalInstructionVerifier.verify_keywords_frequency,
    "keywords:forbidden_words": IFEvalInstructionVerifier.verify_forbidden_words,
    "keywords:letter_frequency": IFEvalInstructionVerifier.verify_letter_frequency,
    "keywords:key_sentences": IFEvalInstructionVerifier.verify_key_sentences,
    "detectable_format:number_bullet_lists": IFEvalInstructionVerifier.verify_number_bullet_lists,
    "detectable_format:json_format": IFEvalInstructionVerifier.verify_json_format,
    "detectable_format:multiple_sections": IFEvalInstructionVerifier.verify_multiple_sections,
    "detectable_format:constrained_response": IFEvalInstructionVerifier.verify_constrained_response,
    "change_case:english_capital": IFEvalInstructionVerifier.verify_english_capital,
    "change_case:english_lowercase": IFEvalInstructionVerifier.verify_english_lowercase,
    "length_constraints:number_paragraphs": IFEvalInstructionVerifier.verify_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": IFEvalInstructionVerifier.verify_nth_paragraph_first_word,
    "startend:end_checker": IFEvalInstructionVerifier.verify_end_checker,
    "startend:quotation": IFEvalInstructionVerifier.verify_quotation,
    "language:response_language": IFEvalInstructionVerifier.verify_response_language,
}


class IFEval(DeepEvalBaseBenchmark):
    """
    IFEval (Instruction Following Evaluation) benchmark implementation.

    IFEval is a benchmark for evaluating instruction-following capabilities of language models.
    It tests various aspects of instruction following including format compliance, constraint
    adherence, output structure requirements, and specific instruction types.

    Based on the original IFEval paper: https://arxiv.org/abs/2311.07911
    and implementation: https://github.com/google-research/google-research/tree/master/instruction_following_eval
    """

    def __init__(
        self,
        n_problems: Optional[int] = None,
        verbose_mode: bool = False,
        **kwargs,
    ):
        import pandas as pd

        from deepeval.scorer import Scorer

        super().__init__(**kwargs)
        self.scorer = Scorer()
        self.n_problems = n_problems
        self.verbose_mode = verbose_mode
        self.predictions: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.instruction_breakdown = None

    def evaluate(self, model: DeepEvalBaseLLM, *args, **kwargs) -> IFEvalResult:
        import pandas as pd

        with capture_benchmark_run("IFEval", self.n_problems or "all"):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            instruction_results = {}

            goldens = self.load_benchmark_dataset()
            if self.n_problems and self.n_problems < len(goldens):
                goldens = goldens[: self.n_problems]

            overall_total_predictions = len(goldens)

            for idx, golden in enumerate(
                tqdm(goldens, desc=f"Processing {len(goldens)} IFEval problems")
            ):
                prediction, score, instruction_scores = self.predict(
                    model, golden
                )
                if score:
                    overall_correct_predictions += 1

                predictions_row.append((golden.input, prediction, score))

                for (
                    instruction_id,
                    instruction_score,
                ) in instruction_scores.items():
                    if instruction_id not in instruction_results:
                        instruction_results[instruction_id] = {
                            "correct": 0,
                            "total": 0,
                        }
                    instruction_results[instruction_id]["total"] += 1
                    if instruction_score:
                        instruction_results[instruction_id]["correct"] += 1

                if self.verbose_mode:
                    self.print_verbose_logs(
                        idx, golden.input, prediction, score, instruction_scores
                    )

            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall IFEval Accuracy: {overall_accuracy:.4f}")

            instruction_accuracies = {}
            for instruction_id, results in instruction_results.items():
                accuracy = results["correct"] / results["total"]
                instruction_accuracies[instruction_id] = accuracy
                print(
                    f"Instruction '{instruction_id}' Accuracy: {accuracy:.4f}"
                )
            predictions: pd.DataFrame = pd.DataFrame(
                predictions_row,
                columns=[
                    "Input",
                    "Prediction",
                    "All_Instructions_Correct",
                ],
            )
            self.predictions = predictions
            self.overall_score = overall_accuracy
            self.instruction_breakdown = instruction_accuracies

            return IFEvalResult(
                overall_accuracy=overall_accuracy,
                instruction_breakdown=instruction_accuracies,
                predictions=predictions,
            )

    def predict(
        self, model: DeepEvalBaseLLM, golden: Golden
    ) -> Tuple[str, bool, Dict[str, bool]]:
        """
        Generate prediction for a single IFEval test case and verify instruction compliance.

        Args:
            model: The language model to evaluate
            golden: The golden test case

        Returns:
            Tuple of (prediction, overall_score, instruction_scores)
        """
        try:
            res: StringSchema = model.generate(
                prompt=golden.input, schema=StringSchema
            )
            prediction = res.answer
        except (TypeError, AttributeError):
            res = model.generate(golden.input)
            prediction = str(res)

        instruction_scores = {}
        all_instructions_passed = True

        metadata = golden.additional_metadata or {}
        instruction_ids = metadata.get("instruction_ids", [])
        kwargs_list = metadata.get("kwargs_list", [])

        for i, instruction_id in enumerate(instruction_ids):
            kwargs = kwargs_list[i] if i < len(kwargs_list) else {}
            passed, reason = (
                IFEvalInstructionVerifier.verify_instruction_compliance(
                    prediction, instruction_id, kwargs
                )
            )
            instruction_scores[instruction_id] = passed
            if not passed:
                all_instructions_passed = False

        return prediction, all_instructions_passed, instruction_scores

    def load_benchmark_dataset(self) -> List[Golden]:
        """
        Load IFEval dataset.

        Returns:
            List of Golden test cases
        """
        from datasets import load_dataset

        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("google/IFEval")
            self.dataset = dataset

        goldens: List[Golden] = []

        train_data = dataset["train"]

        for data in train_data:
            prompt = data.get("prompt", "")
            instruction_id_list = data.get("instruction_id_list", [])
            kwargs = data.get("kwargs", [])

            golden = Golden(input=prompt, expected_output="")
            golden.additional_metadata = {
                "instruction_ids": instruction_id_list,
                "kwargs_list": kwargs,
            }

            goldens.append(golden)

        return goldens

    def print_verbose_logs(
        self,
        idx: int,
        input: str,
        prediction: str,
        score: bool,
        instruction_scores: Dict[str, bool],
    ) -> str:
        """
        Print verbose logs for debugging and analysis.

        Args:
            idx: Problem index
            input: Input instruction
            prediction: Model prediction
            score: Overall score (True if all instructions passed)
            instruction_scores: Individual instruction scores

        Returns:
            Formatted verbose log string
        """
        steps = [
            f"Input:\n{input}",
            f"Overall Score: {score}\nPrediction: {prediction[:200]}{'...' if len(prediction) > 200 else ''}",
            "Instruction Breakdown:\n"
            + "\n".join(
                [
                    f"  {inst}: {'✓' if passed else '✗'}"
                    for inst, passed in instruction_scores.items()
                ]
            ),
        ]
        verbose_logs = ""
        for i in range(len(steps) - 1):
            verbose_logs += steps[i]

            if i < len(steps) - 2:
                verbose_logs += " \n \n"

        if self.verbose_mode:
            print("*" * 50)
            print(f"Problem {idx + 1}")
            print("*" * 50)
            print("")
            print(verbose_logs + f"\n \n{steps[-1]}")
            print("")
            print("=" * 70)

        return verbose_logs
