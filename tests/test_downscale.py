import pytest
from PIL import Image

from swo.downscale import fit_to_budget

BUDGETS = [1, 2_000, 12_500, 100_000, 800_000]
SIZES = [(1920, 1080), (4000, 3000), (640, 480), (333, 777), (10_000, 1), (1, 10_000)]


def image(width: int, height: int, mode: str = "RGB") -> Image.Image:
    return Image.new(mode, (width, height))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("budget", BUDGETS)
def test_never_exceeds_the_budget(size, budget):
    result = fit_to_budget(image(*size), budget)
    assert result.width >= 1 and result.height >= 1
    assert result.width * result.height <= budget


def test_image_within_budget_is_returned_unchanged():
    original = image(100, 100)
    assert fit_to_budget(original, 20_000) is original


def test_never_upscales():
    assert fit_to_budget(image(50, 40), 10**6).size == (50, 40)


def test_preserves_aspect_ratio():
    result = fit_to_budget(image(1600, 900), 100_000)
    assert result.width / result.height == pytest.approx(1600 / 900, rel=0.01)


def test_uses_nearly_all_of_the_budget():
    # A correct scale factor lands just under the budget, not far below it.
    result = fit_to_budget(image(1600, 900), 100_000)
    assert result.width * result.height > 0.99 * 100_000


def test_extreme_aspect_ratio_still_respects_the_budget():
    # Flooring 10000x1 would give 4472x1, which the long-side cap has to trim.
    result = fit_to_budget(image(10_000, 1), 2_000)
    assert result.size == (2_000, 1)


@pytest.mark.parametrize("budget", [0, -1])
def test_rejects_non_positive_budget(budget):
    with pytest.raises(ValueError):
        fit_to_budget(image(10, 10), budget)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_handles_common_image_modes(mode):
    result = fit_to_budget(image(400, 300, mode), 10_000)
    assert result.mode == mode
    assert result.width * result.height <= 10_000
