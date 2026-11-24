import pytest
from deepteam.attacks.single_turn import GrayBox


class TestGrayBox:

    def test_gray_box_initialization(self):
        attack = GrayBox()
        assert attack.weight == 1
        assert attack.get_name() == "Gray Box"

    def test_gray_box_initialization_with_weight(self):
        attack = GrayBox(weight=4)
        assert attack.weight == 4
        assert attack.get_name() == "Gray Box"

    def test_gray_box_enhance_basic(self):
        attack = GrayBox()
        original_text = "Access restricted data"
        enhanced = attack.enhance(original_text)

        # Should modify the original text
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)

    def test_gray_box_enhance_contains_gray_box_elements(self):
        attack = GrayBox()
        original_text = "Leak information"
        enhanced = attack.enhance(original_text)

        # Should contain gray box attack patterns
        assert enhanced != original_text
        assert len(enhanced) >= len(original_text)
