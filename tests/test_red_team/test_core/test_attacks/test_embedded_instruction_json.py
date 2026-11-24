import json
import pytest
from deepteam.attacks.single_turn import EmbeddedInstructionJSON


def extract_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text")
    return json.loads(text[start:])


class TestEmbeddedInstructionJSON:

    def test_embedded_instruction_json_initialization(self):
        attack = EmbeddedInstructionJSON()
        assert attack.weight == 1
        assert attack.get_name() == "EmbeddedInstructionJSON"

    def test_embedded_instruction_json_initialization_with_weight(self):
        attack = EmbeddedInstructionJSON(weight=3)
        assert attack.weight == 3

    def test_embedded_instruction_json_enhance_basic(self):
        attack = EmbeddedInstructionJSON(target_chars=500)
        payload = "test payload"
        enhanced = attack.enhance(payload)

        assert payload in enhanced
        assert len(enhanced) >= 500

    def test_embedded_instruction_json_reaches_target_chars(self):
        target_chars = 1000
        attack = EmbeddedInstructionJSON(target_chars=target_chars)
        enhanced = attack.enhance("attack")

        assert len(enhanced) >= target_chars

    def test_embedded_instruction_json_contains_main_note(self):
        attack = EmbeddedInstructionJSON(target_chars=500)
        payload = "important instruction"
        enhanced = attack.enhance(payload)

        parsed = extract_json(enhanced)

        def find_main_note(obj):
            if isinstance(obj, dict):
                if "main_note" in obj:
                    return obj["main_note"]
                for v in obj.values():
                    found = find_main_note(v)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = find_main_note(item)
                    if found is not None:
                        return found
            return None

        assert find_main_note(parsed) == payload

    def test_embedded_instruction_json_attack_position_start(self):
        attack = EmbeddedInstructionJSON(
            target_chars=500, attack_position="start"
        )
        enhanced = attack.enhance("payload")
        parsed = extract_json(enhanced)

        first_key = next(iter(parsed.keys()))
        assert first_key == "deep_context"

    def test_embedded_instruction_json_attack_position_middle(self):
        attack = EmbeddedInstructionJSON(
            target_chars=500, attack_position="middle"
        )
        enhanced = attack.enhance("payload")
        parsed = extract_json(enhanced)

        keys = list(parsed.keys())
        assert "deep_context" in keys
        assert keys.index("deep_context") != 0
        assert keys.index("deep_context") != len(keys) - 1

    def test_embedded_instruction_json_attack_position_end(self):
        attack = EmbeddedInstructionJSON(
            target_chars=500, attack_position="end"
        )
        enhanced = attack.enhance("payload")
        parsed = extract_json(enhanced)

        last_key = list(parsed.keys())[-1]
        assert last_key == "deep_context"

    def test_embedded_instruction_json_custom_template(self):
        custom_json = {
            "tool": "search",
            "results": [{"title": "doc", "snippet": "text"}],
        }
        attack = EmbeddedInstructionJSON(
            json_template=custom_json,
            target_chars=500,
        )
        enhanced = attack.enhance("payload")
        parsed = extract_json(enhanced)

        assert "tool" in parsed
        assert "results" in parsed

    def test_embedded_instruction_json_deterministic(self):
        attack = EmbeddedInstructionJSON(target_chars=700)
        payload = "deterministic payload"

        enhanced_1 = attack.enhance(payload)
        enhanced_2 = attack.enhance(payload)

        assert enhanced_1 == enhanced_2

    def test_embedded_instruction_json_invalid_target_chars(self):
        with pytest.raises(ValueError):
            EmbeddedInstructionJSON(target_chars=10)

    def test_embedded_instruction_json_invalid_target_depth(self):
        with pytest.raises(ValueError):
            EmbeddedInstructionJSON(target_depth=0)

    def test_embedded_instruction_json_invalid_json_template(self):
        with pytest.raises(ValueError):
            EmbeddedInstructionJSON(json_template="not a dict")
