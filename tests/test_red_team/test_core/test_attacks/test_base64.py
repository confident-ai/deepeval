import pytest
import base64
from deepteam.attacks.single_turn import Base64


class TestBase64:

    def test_base64_initialization(self):
        attack = Base64()
        assert attack.weight == 1
        assert attack.get_name() == "Base64"

    def test_base64_initialization_with_weight(self):
        attack = Base64(weight=5)
        assert attack.weight == 5
        assert attack.get_name() == "Base64"

    def test_base64_enhance(self):
        attack = Base64()
        original_text = "Hello, world!"
        enhanced = attack.enhance(original_text)

        # Verify it's base64 encoded
        decoded = base64.b64decode(enhanced).decode()
        assert decoded == original_text

    def test_base64_enhance_special_characters(self):
        attack = Base64()
        original_text = "!@#$%^&*()_+"
        enhanced = attack.enhance(original_text)
        decoded = base64.b64decode(enhanced).decode()
        assert decoded == original_text

    def test_base64_enhance_unicode(self):
        attack = Base64()
        original_text = "Hello 世界 🌍"
        enhanced = attack.enhance(original_text)
        decoded = base64.b64decode(enhanced).decode()
        assert decoded == original_text
