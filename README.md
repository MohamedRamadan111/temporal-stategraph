# TemporalStateGraph

**Calibrated Decay for Stale-Memory Suppression in Long-Horizon AI Agents**

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

Long-horizon AI agents accumulate contradictory and outdated facts over time. TemporalStateGraph (TG) addresses this by assigning **calibrated exponential decay rates** to stored facts based on their semantic type:

- 🪪 **Identity facts** (name, core identity) → half-life ~578 days  
- 💼 **Employer / Goals** → half-life ~6 days  
- 🥗 **Preferences** → half-life ~2 days  
- ✅ **Tasks** → half-life ~23 hours  
- 😐 **Mood** → half-life ~10 hours  

When a fact is contradicted, its decay accelerates 3×. Facts below a retrieval threshold are suppressed without deletion, preserving historical recall while hiding stale state.

### Key Results

| System | Accuracy | Stale Rate |
|--------|----------|-----------|
| **TG (Ours)** | **84.8%** | **15.2%** |
| StaticMemory | 48.3% | 52.4% |
| SlidingWindow | 48.3% | 52.4% |
| RecencyMemory | 17.5% | 75.9% |

*N=63 scenarios, 315 query evaluations. p < 10⁻²⁰ vs. StaticMemory, Cohen's d = 2.61.*

---

## Installation

```bash
git clone https://github.com/ANONYMIZED/temporal-stategraph
cd temporal-stategraph
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, numpy, scipy, matplotlib, openai

---

## Quickstart

```python
from temporal_stategraph import TemporalStateGraph

# Initialize memory
memory = TemporalStateGraph()

# Add facts (key, value, type)
memory.add_fact("employer", "Google", "goal")       # t=0h
memory.add_fact("diet", "vegetarian", "preference") # t=5h
memory.add_fact("mood", "stressed", "mood")         # t=10h
memory.add_fact("employer", "Microsoft", "goal")    # t=40h — triggers contradiction decay
memory.advance_time(97)  # advance to query time

# Retrieve context
context = memory.get_context()
print(context)
# [2.51] employer: Microsoft
# [2.00] diet: vegetarian
# [0.84] employer: Google   ← still present but lower score
# mood entry suppressed (score < 0.15)
```

---

## Architecture

```
Event Stream
     │
     ▼
┌─────────────────────────────┐
│     TemporalStateGraph      │
│                             │
│  fact = (key, value,        │
│          t_add, λ, α, r)    │
│                             │
│  S(f,t) = exp(-λ·Δt)·α·    │
│           (1+log(1+r))      │
│                             │
│  Contradiction: λ ×= 3.0   │
│  Threshold:   S > 0.15     │
└─────────────────────────────┘
     │
     ▼
Ranked Context Window → LLM
```

### Decay Types

| Type | λ (per hour) | Half-life |
|------|-------------|----------|
| identity | 5×10⁻⁵ | 578 days |
| goal | 0.005 | 6 days |
| preference | 0.015 | 2 days |
| task | 0.030 | 23 hours |
| mood | 0.070 | 10 hours |
| noise | 0.040 | 17 hours |

---

## Benchmark

Run the full evaluation benchmark:

```bash
# Set API key
export OPENROUTER_API_KEY=sk-or-v1-...

# Run benchmark (uses cache — skips already-completed calls)
python benchmark/llm_benchmark.py

# Analyze results
python benchmark/analysis.py

# Generate all figures
python benchmark/build_figures.py
```

**Expected runtime:** ~45 min for fresh run (rate-limited to 1.1 req/sec)  
**Cached runs:** ~30 sec (JSON cache reused automatically)

### Benchmark Structure

```
63 scenarios × 4 systems × 5 queries = 1,260 LLM calls total
Seeds: {42, 1337}
Judge: GPT-4o-mini (temperature=0)
```

---

## Project Structure

```
temporal-stategraph/
├── temporal_stategraph/
│   ├── __init__.py
│   ├── core.py              # TemporalStateGraph class
│   ├── baselines.py         # StaticMemory, RecencyMemory, SlidingWindow
│   └── decay_rates.py       # DECAY_RATES config
├── benchmark/
│   ├── llm_benchmark.py     # Main benchmark runner
│   ├── analysis.py          # Statistical analysis
│   ├── build_figures.py     # Figure generation
│   └── scenarios.py         # Scenario generation
├── paper/
│   └── main.pdf             # Submitted paper
├── figures/                 # All publication figures
├── results/
│   ├── final_results.json   # Per-scenario metrics
│   └── detailed_outputs.json# Per-query answers + judgments
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reproducing Paper Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run benchmark (or use cached results in results/)
export OPENROUTER_API_KEY=your_key_here
python benchmark/llm_benchmark.py

# 3. Generate tables and statistics
python benchmark/analysis.py
# → prints Tables 1-4 from paper

# 4. Generate all figures  
python benchmark/build_figures.py
# → saves to figures/ (PNG + PDF)
```

All cached results are included in `results/` for reproducibility without API access.

---

## Citation

```bibtex
@article{temporal_stategraph_2025,
  title={Temporal StateGraph: Calibrated Decay for Stale-Memory 
         Suppression in Long-Horizon AI Agents},
  author={Anonymous},
  year={2025},
  note={Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome. Please:
1. Open an issue describing the change
2. Fork and create a feature branch
3. Ensure all tests pass: `python -m pytest tests/`
4. Submit a pull request with clear description
