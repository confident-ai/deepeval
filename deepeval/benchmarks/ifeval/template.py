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
        # Look for common response indicators
        response_indicators = [
            "Response:",
            "Answer:",
            "Output:",
            "Result:"
        ]

        for indicator in response_indicators:
            if indicator in text:
                parts = text.split(indicator, 1)
                if len(parts) > 1:
                    return parts[1].strip()

        # If no indicator found, return the whole text
        return text.strip()

    @staticmethod
    def validate_format_compliance(response: str, expected_format: str) -> bool:
        """
        Validate if a response follows the expected format.

        Args:
            response: The model's response
            expected_format: The expected format description

        Returns:
            True if format is compliant, False otherwise
        """
        # This is a basic implementation - can be extended based on specific format requirements
        if "json" in expected_format.lower():
            try:
                import json
                json.loads(response)
                return True
            except Exception:
                return False
        elif "list" in expected_format.lower():
            # Check if response looks like a list
            lines = response.strip().split('\n')
            return len(lines) > 1 or response.strip().startswith('-') or response.strip().startswith('*')
        else:
            # For other formats, basic validation
            return len(response.strip()) > 0

    @staticmethod
    def check_constraint_adherence(response: str, constraints: list) -> bool:
        """
        Check if a response adheres to given constraints.

        Args:
            response: The model's response
            constraints: List of constraint descriptions

        Returns:
            True if all constraints are adhered to, False otherwise
        """
        # Basic constraint checking - can be extended based on specific constraint types
        for constraint in constraints:
            if "length" in constraint.lower():
                # Check length constraints
                if "max" in constraint.lower():
                    max_length = int(''.join(filter(str.isdigit, constraint)))
                    if len(response) > max_length:
                        return False
                elif "min" in constraint.lower():
                    min_length = int(''.join(filter(str.isdigit, constraint)))
                    if len(response) < min_length:
                        return False
            elif "contains" in constraint.lower():
                # Check if response contains required elements
                required_elements = constraint.split(
                    "contains")[1].strip().split(",")
                for element in required_elements:
                    if element.strip() not in response:
                        return False
            elif "not_contains" in constraint.lower():
                # Check if response doesn't contain forbidden elements
                forbidden_elements = constraint.split(
                    "not_contains")[1].strip().split(",")
                for element in forbidden_elements:
                    if element.strip() in response:
                        return False

        return True
