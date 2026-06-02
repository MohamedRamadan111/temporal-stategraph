# TemporalStateGraph (TG)

**Calibrated Decay for Stale-Memory Suppression in Long-Horizon AI Agents**

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

Long-horizon AI agents operating over extended periods naturally accumulate contradictory, obsolete, and stale facts, causing severe context-window pollution and retrieval failures.

TemporalStateGraph (TG) introduces a lightweight memory layer that assigns domain-calibrated exponential decay rates to stored facts according to their semantic type.

| Memory Type | Approximate Half-Life |
|------------|----------------------|
| Identity Facts | ~578 days |
| Employer / Strategic Goals | ~6 days |
| User Preferences | ~2 days |
| Task Statuses | ~23 hours |
| Mood / Emotional States | ~10 hours |

---

## Key Contributions

### Contradiction-Accelerated Decay

When an incoming fact contradicts an existing memory record, TG dynamically scales the previous record's decay rate:

$$
\lambda_{new} = \gamma \lambda_{old}
$$

where:

$$
\gamma = 3.0
$$

This allows outdated information to fade naturally without explicit deletion.

### Intentional Forgetting

Memories whose temporal strength falls below:

$$
\theta = 0.15
$$

are automatically excluded from prompt construction, reducing stale-memory pollution.

---

## Empirical Results

| Memory System Architecture | Accuracy ↑ | Stale Memory Rate ↓ |
|---------------------------|------------|---------------------|
| **TemporalStateGraph (Ours)** | **84.8%** | **15.2%** |
| StaticMemory | 48.3% | 52.4% |
| SlidingWindow | 48.3% | 52.4% |
| RecencyMemory | 17.5% | 75.9% |

TG achieves a +36.5 percentage-point improvement over conventional conversational memory buffers with a large effect size (Cohen's d = 2.61).

---

## Mathematical Framework

$$
S(f,t)
=
\exp\Bigl(-\lambda_f (t-t_{add})\Bigr)
\cdot
\alpha_f
\cdot
\Bigl(1+\log(1+r_f)\Bigr)
$$

where:

- $\lambda_f$ : decay coefficient
- $\alpha_f$ : importance weight
- $r_f$ : reinforcement count
- $t-t_{add}$ : elapsed time

---

## Installation

```bash
git clone https://github.com/MohamedRamadan111/temporal-stategraph
cd temporal-stategraph
pip install -r requirements.txt
```

## Quick Start

```python
from temporal_stategraph import TemporalStateGraph

memory = TemporalStateGraph()

memory.add_fact("employer", "Google", "goal")
memory.add_fact("diet", "vegetarian", "preference")

memory.advance_time(40)

memory.add_fact("employer", "Microsoft", "goal")

memory.advance_time(57)

print(memory.get_context())
```

---

## Reproducing Results

```bash
python benchmark/llm_benchmark.py
python benchmark/analysis.py
python benchmark/build_figures.py
```

---

## Citation

```bibtex
@article{temporal_stategraph_2026,
  title={Temporal StateGraph: Calibrated Decay for Stale-Memory Suppression in Long-Horizon AI Agents},
  author={Anonymous},
  year={2026},
  note={Preprint}
}
```

---

## License

Released under the MIT License.
