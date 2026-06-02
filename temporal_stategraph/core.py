"""
core.py
TemporalStateGraph — main memory class.

Architecture:
  Each fact f = (key, value, t_add, lambda, importance, reinforcement)
  S(f, t) = exp(-λ * Δt) * α * (1 + log(1 + r))
  Facts with S < threshold are suppressed from context.
  Contradicted facts receive λ *= CONTRADICTION_MULTIPLIER.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .decay_rates import (
    DECAY_RATES, DEFAULT_THRESHOLD,
    CONTRADICTION_MULTIPLIER, MAX_CONTEXT_FACTS
)


@dataclass
class TemporalFact:
    key:            str
    value:          Any
    timestamp:      float          # simulated time (hours from start)
    lambda_decay:   float          # per-hour decay rate
    importance:     float = 1.0
    reinforcement:  int   = 1

    def score(self, now: float) -> float:
        """
        S(f, t) = exp(-λ · Δt) · α · (1 + log(1 + r))
        """
        dt = max(now - self.timestamp, 0.0)
        decay  = math.exp(-self.lambda_decay * dt)
        reinf  = 1.0 + math.log1p(self.reinforcement)
        return decay * reinf * self.importance

    def is_active(self, now: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
        return self.score(now) > threshold


class TemporalStateGraph:
    """
    Memory architecture with per-type calibrated decay.

    Usage:
        mem = TemporalStateGraph()
        mem.add_fact("employer", "Google",    "goal")
        mem.add_fact("diet",     "vegan",     "preference")
        mem.add_fact("mood",     "stressed",  "mood")
        mem.advance_time(50)
        mem.add_fact("employer", "Microsoft", "goal")  # triggers contradiction decay
        mem.advance_time(47)
        ctx = mem.get_context()   # returns ranked active facts
    """

    def __init__(
        self,
        decay_rates: Optional[dict] = None,
        threshold: float = DEFAULT_THRESHOLD,
        contradiction_multiplier: float = CONTRADICTION_MULTIPLIER,
        max_context_facts: int = MAX_CONTEXT_FACTS,
    ):
        self.decay_rates              = decay_rates or DECAY_RATES.copy()
        self.threshold                = threshold
        self.contradiction_multiplier = contradiction_multiplier
        self.max_context_facts        = max_context_facts
        self.now: float               = 0.0
        self.facts: List[TemporalFact] = []

    # ── Time management ──────────────────────────────────────────────────────

    def advance_time(self, hours: float) -> None:
        """Advance the internal clock by `hours`."""
        self.now += hours

    def set_time(self, t: float) -> None:
        """Set absolute internal time."""
        self.now = t

    # ── Fact management ───────────────────────────────────────────────────────

    def add_fact(
        self,
        key: str,
        value: Any,
        fact_type: str = "goal",
        importance: float = 1.0,
    ) -> TemporalFact:
        """
        Insert a fact.
        - If a fact with the same key but different value exists,
          its decay rate is multiplied by contradiction_multiplier.
        - If a fact with the same key and same value exists,
          its reinforcement counter increments.
        """
        lam = self.decay_rates.get(fact_type, 0.01)

        # Contradiction detection
        for f in self.facts:
            if f.key == key and f.value != value:
                f.lambda_decay *= self.contradiction_multiplier

        # Reinforcement detection
        for f in self.facts:
            if f.key == key and f.value == value:
                f.reinforcement += 1

        fact = TemporalFact(
            key=key,
            value=value,
            timestamp=self.now,
            lambda_decay=lam,
            importance=importance,
        )
        self.facts.append(fact)
        return fact

    def get_context(self, max_facts: Optional[int] = None) -> str:
        """
        Return active facts sorted by temporal score (descending).
        Format: '[score] key: value'
        Facts below threshold are suppressed without deletion.
        """
        k = max_facts or self.max_context_facts
        active = [
            (f, f.score(self.now))
            for f in self.facts
            if f.is_active(self.now, self.threshold)
        ]
        active.sort(key=lambda x: -x[1])
        lines = [f"[{s:.2f}] {f.key}: {f.value}" for f, s in active[:k]]
        return "\n".join(lines) if lines else "(no active memory)"

    def get_active_facts(self) -> List[TemporalFact]:
        """Return list of currently active TemporalFact objects."""
        return [f for f in self.facts if f.is_active(self.now, self.threshold)]

    def get_all_facts(self) -> List[TemporalFact]:
        """Return all stored facts including suppressed ones."""
        return list(self.facts)

    def __repr__(self) -> str:
        n_active = len(self.get_active_facts())
        return (
            f"TemporalStateGraph(t={self.now:.1f}h, "
            f"facts={len(self.facts)}, active={n_active})"
        )
