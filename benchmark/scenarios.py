"""
scenarios.py — Scenario generation for the agent memory benchmark.
Each scenario: 11 events over 73h + 5 queries at t≥97h.
"""
from __future__ import annotations
import random
from typing import List, Dict, Tuple

NAMES     = ["Ahmed","Lina","Omar","Sara","Karim","Dina","Youssef","Nour"]
COMPANIES = ["Google","Microsoft","Amazon","OpenAI","Meta","Apple","Stripe","Notion","Figma","Anthropic"]
DIETS     = ["vegetarian","vegan","keto","pescatarian","omnivore"]
NOISES    = ["watched Netflix","bought headphones","ordered coffee","visited bookstore",
             "changed wallpaper","liked football","watched TV"]

def build_scenario(seed_val: int, scenario_idx: int) -> Dict:
    rng = random.Random(seed_val * 1000 + scenario_idx)
    name     = rng.choice(NAMES)
    old_co   = rng.choice(COMPANIES)
    new_co   = rng.choice([c for c in COMPANIES if c != old_co])
    old_diet = rng.choice(DIETS)
    new_diet = rng.choice([d for d in DIETS if d != old_diet])
    noise    = [rng.choice(NOISES) for _ in range(4)]

    events = [
        (0,  "name",     name,     "identity",   1.5),
        (2,  "employer", old_co,   "goal",        1.5),
        (5,  "diet",     old_diet, "preference",  1.2),
        (8,  "mood",     "stressed about presentation", "mood", 0.8),
        (10, "noise1",   noise[0], "noise",       0.3),
        (15, "project",  "Q4 report pending",    "task",  1.3),
        (20, "noise2",   noise[1], "noise",       0.3),
        (28, "project",  "Q4 report completed",  "task",  1.4),
        (35, "diet",     new_diet, "preference",  1.4),
        (42, "employer", new_co,   "goal",        1.7),
        (55, "noise3",   noise[2], "noise",       0.3),
    ]

    queries = [
        {"question": f"Where does {name} currently work?",
         "expected": new_co,    "stale": old_co},
        {"question": f"What is {name}'s current diet?",
         "expected": new_diet,  "stale": old_diet},
        {"question": "Is the Q4 report still pending?",
         "expected": "completed", "stale": "pending"},
        {"question": f"How is {name} feeling right now?",
         "expected": "no current mood information", "stale": "stressed"},
        {"question": f"What company did {name} work at BEFORE the current one?",
         "expected": old_co, "stale": new_co},
    ]
    return {"events": events, "queries": queries, "name": name}
