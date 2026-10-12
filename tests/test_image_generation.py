from types import SimpleNamespace

import numpy as np
from PIL import Image

from ink_ai.brain.image_generation import LocalImageGenerator


class FakeFeatureExtractor:
    def __call__(self, *, images, return_tensors):
        assert len(images) == 1
        assert return_tensors == "pt"
        return SimpleNamespace(pixel_values="clip input")


class FakeSafetyChecker:
    def __init__(self, flags):
        self.flags = flags
        self.clip_input = None
        self.images = None

    def __call__(self, *, clip_input, images):
        self.clip_input = clip_input
        self.images = images
        return images, self.flags


def test_safety_checker_receives_generated_image_and_returns_flag():
    generator = LocalImageGenerator()
    checker = FakeSafetyChecker([True])
    generator.feature_extractor = FakeFeatureExtractor()
    generator.safety_checker = checker
    image = Image.new("RGB", (4, 4))

    assert generator._check_nsfw([image]) == [True]
    assert checker.clip_input == "clip input"
    assert isinstance(checker.images, np.ndarray)
    assert checker.images.shape == (1, 4, 4, 3)


def test_safety_checker_missing_result_fails_explicitly():
    generator = LocalImageGenerator()
    generator.feature_extractor = FakeFeatureExtractor()
    generator.safety_checker = FakeSafetyChecker(None)

    try:
        generator._check_nsfw([Image.new("RGB", (4, 4))])
    except RuntimeError as error:
        assert "geçerli sonuç" in str(error)
    else:
        raise AssertionError("Missing safety-check results must be reported.")
