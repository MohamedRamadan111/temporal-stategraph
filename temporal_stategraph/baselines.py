"""
baselines.py
Three baselines for comparison against TemporalStateGraph.
"""
from __future__ import annotations
import math
from typing import Any, List, Optional
from .decay_rates import DECAY_RATES, DEFAULT_THRESHOLD, MAX_CONTEXT_FACTS


class StaticMemory:
    """
    FIFO buffer — no temporal scoring.
    Analogue: LangChain ConversationBufferMemory.
    """
    name = "StaticMemory"

    def __init__(self, max_size: int = 20):
        self.facts: list = []
        self.max_size = max_size
        self.now: float = 0.0

    def advance_time(self, h): self.now += h
    def set_time(self, t):    self.now = t

    def add_fact(self, key, value, fact_type="goal", importance=1.0):
        self.facts.append((key, value))
        if len(self.facts) > self.max_size:
            self.facts.pop(0)

    def get_context(self, max_facts=None) -> str:
        k = max_facts or MAX_CONTEXT_FACTS
        recent = self.facts[-k:]
        return "\n".join(f"{k_}: {v}" for k_, v in recent)


class RecencyMemory:
    """
    Uniform recency weighting — same λ for all fact types.
    Represents uncalibrated recency-based retrieval (e.g., naive Mem0-style).
    """
    name = "RecencyMemory"
    UNIFORM_LAMBDA = 0.020   # per hour

    def __init__(self, uniform_lambda: float = 0.020):
        self.lam = uniform_lambda
        self.facts: list = []
        self.now: float = 0.0

    def advance_time(self, h): self.now += h
    def set_time(self, t):    self.now = t

    def add_fact(self, key, value, fact_type="goal", importance=1.0):
        self.facts.append({"key": key, "value": value, "t": self.now})

    def get_context(self, max_facts=None) -> str:
        k = max_facts or MAX_CONTEXT_FACTS
        scored = []
        for f in self.facts:
            dt = max(self.now - f["t"], 0.0)
            s  = math.exp(-self.lam * dt)
            if s > 0.05:
                scored.append((f, s))
        scored.sort(key=lambda x: -x[1])
        lines = [f"{f['key']}: {f['value']}" for f, _ in scored[:k]]
        return "\n".join(lines)


class SlidingWindow:
    """
    Most-recent k facts — no temporal scoring.
    Represents sliding-window context management.
    """
    name = "SlidingWindow"

    def __init__(self, window_size: int = MAX_CONTEXT_FACTS):
        self.facts: list = []
        self.window = window_size
        self.now: float = 0.0

    def advance_time(self, h): self.now += h
    def set_time(self, t):    self.now = t

    def add_fact(self, key, value, fact_type="goal", importance=1.0):
        self.facts.append((key, value))

    def get_context(self, max_facts=None) -> str:
        k = max_facts or self.window
        recent = self.facts[-k:]
        return "\n".join(f"{k_}: {v}" for k_, v in recent)
