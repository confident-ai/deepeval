class IFEvalTemplate:
    """
    Template utilities for IFEval benchmark.

    Provides methods for formatting instructions and processing responses
    for the IFEval instruction following evaluation benchmark.

    Based on the original IFEval implementation from Google Research.
    """

    @staticmethod
    def format_instruction(instruction: str) -> str:
        """
        Format an instruction for the IFEval benchmark.

        Args:
            instruction: The raw instruction text

        Returns:
            Formatted instruction string
        """
        return f"Instruction: {instruction}\n\nResponse:"

    @staticmethod
    def extract_response(text: str) -> str:
        """
        Extract the response part from a model's output.

        Args:
            text: The model's output text

        Returns:
            Extracted response string
        """
        response_indicators = ["Response:", "Answer:", "Output:", "Result:"]

        for indicator in response_indicators:
            if indicator in text:
                parts = text.split(indicator, 1)
                if len(parts) > 1:
                    return parts[1].strip()

        return text.strip()

    @staticmethod
    def get_instruction_category(instruction_id: str) -> str:
        """
        Get the category of an instruction based on its ID.

        Args:
            instruction_id: The instruction ID (e.g., "punctuation:no_comma")

        Returns:
            The instruction category (e.g., "punctuation")
        """
        return (
            instruction_id.split(":")[0] if ":" in instruction_id else "unknown"
        )

    @staticmethod
    def get_instruction_description(instruction_id: str) -> str:
        """
        Get a human-readable description of an instruction.

        Args:
            instruction_id: The instruction ID

        Returns:
            Human-readable description
        """
        descriptions = {
            "punctuation:no_comma": "No commas allowed",
            "punctuation:no_period": "No periods allowed",
            "punctuation:no_question_mark": "No question marks allowed",
            "punctuation:no_exclamation_mark": "No exclamation marks allowed",
            "length_constraints:number_words": "Word count constraint",
            "length_constraints:number_characters": "Character count constraint",
            "length_constraints:number_sentences": "Sentence count constraint",
            "detectable_format:json": "Must be valid JSON format",
            "detectable_format:list": "Must be in list format",
            "detectable_format:number_bullets": "Must have specified number of bullet points",
            "detectable_format:number_highlighted_sections": "Must have specified number of highlighted sections",
            "detectable_content:keyword_frequency": "Must contain keyword with specified frequency",
            "detectable_content:forbidden_words": "Must not contain forbidden words",
            "detectable_content:number_placeholders": "Must have specified number of placeholders",
            "detectable_content:postscript": "Must contain postscript marker",
            "detectable_content:first_word": "Must start with specified word",
            "structural_constraints:number_paragraphs": "Must have specified number of paragraphs",
            "structural_constraints:number_sections": "Must have specified number of sections",
            "combination:repeat_prompt": "Must repeat the specified prompt",
        }

        return descriptions.get(
            instruction_id, f"Unknown instruction: {instruction_id}"
        )

    @staticmethod
    def format_verification_report(
        instruction_scores: dict, prediction: str
    ) -> str:
        """
        Format a detailed verification report for verbose output.

        Args:
            instruction_scores: Dictionary mapping instruction IDs to boolean results
            prediction: The model's prediction

        Returns:
            Formatted verification report
        """
        report = "=== IFEval Verification Report ===\n\n"
        report += f"Prediction Length: {len(prediction)} characters, {len(prediction.split())} words\n\n"

        categories = {}
        for instruction_id, passed in instruction_scores.items():
            category = IFEvalTemplate.get_instruction_category(instruction_id)
            if category not in categories:
                categories[category] = []
            categories[category].append((instruction_id, passed))

        for category, instructions in categories.items():
            report += f"--- {category.upper()} ---\n"
            for instruction_id, passed in instructions:
                status = "✓ PASS" if passed else "✗ FAIL"
                description = IFEvalTemplate.get_instruction_description(
                    instruction_id
                )
                report += f"  {status}: {description} ({instruction_id})\n"
            report += "\n"

        return report
