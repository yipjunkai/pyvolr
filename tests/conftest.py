"""Shared pytest fixtures and Hypothesis profile configuration."""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# Two profiles: fast (PR runs) and thorough (nightly).
settings.register_profile(
    "fast",
    max_examples=50,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "thorough",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("fast")
