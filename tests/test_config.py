from __future__ import annotations

import pytest

from tickyantra.config import Settings


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(mode="toy").validate()


def test_adaptive_bounds_are_validated() -> None:
    with pytest.raises(ValueError):
        Settings(adaptive_min_limit=10, adaptive_initial_limit=2).validate()
