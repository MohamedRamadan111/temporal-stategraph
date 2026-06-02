"""
decay_rates.py
Calibrated per-type decay rates for TemporalStateGraph.

λ values are per-hour; half-life = ln(2) / λ.
These are derived from human memory literature and calibrated
against the agent memory benchmark.
"""

DECAY_RATES = {
    "identity":   5e-5,   # half-life ~578 days — name, core identity
    "goal":       0.005,  # half-life ~6 days   — employer, long-term goal
    "preference": 0.015,  # half-life ~2 days   — dietary, lifestyle preferences
    "task":       0.030,  # half-life ~23 hours  — task status, pending items
    "mood":       0.070,  # half-life ~10 hours  — transient emotional state
    "noise":      0.040,  # half-life ~17 hours  — incidental events
}

# Fact below this threshold is suppressed from context window
DEFAULT_THRESHOLD = 0.15

# Contradicted facts decay this many times faster than natural rate
CONTRADICTION_MULTIPLIER = 3.0

# Default max facts returned in context
MAX_CONTEXT_FACTS = 14
