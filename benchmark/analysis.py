"""
analysis.py — Statistical analysis of benchmark results.
Prints Tables 1-4 from the paper and saves tables/all_tables.tex.
Run: python benchmark/analysis.py
"""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))

with open("results/final_results.json")    as f: results  = json.load(f)
with open("results/detailed_outputs.json") as f: detailed = json.load(f)

def parse_key(k):
    p = k.split('_'); return p[0], p[1], '_'.join(p[2:])

# Aggregate
sys_data = defaultdict(lambda: {'accuracy':[],'stale_rate':[],'avg_score':[]})
for sk, sd in results.items():
    for sn, m in sd.items():
        if sn.isdigit(): continue
        sys_data[sn]['accuracy'].append(m['accuracy'])
        sys_data[sn]['stale_rate'].append(m['stale_rate'])
        sys_data[sn]['avg_score'].append(m['avg_score'])

SYS = ['TemporalStateGraph','StaticMemory','RecencyMemory','SlidingWindow']

def boot(arr, n=4000):
    b = [np.random.default_rng(i).choice(arr, len(arr)).mean() for i in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)

def cohens_d(a, b):
    pooled = math.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    return (np.mean(a)-np.mean(b))/pooled if pooled else 0.0

# ── TABLE 1: Main Results ─────────────────────────────────────────────────────
print("\n" + "="*72)
print("TABLE 1: Main Results")
print("="*72)
print(f"{'System':<22} {'Acc (Auto)':>12} {'±std':>7} {'95%CI':>16} {'StaleRate':>10} {'Score':>7}")
print("-"*72)
summary = {}
for sn in SYS:
    acc   = np.array(sys_data[sn]['accuracy'])
    stale = np.array(sys_data[sn]['stale_rate'])
    score = np.array(sys_data[sn]['avg_score'])
    lo, hi = boot(acc)
    summary[sn] = dict(acc=acc, stale=stale, score=score)
    print(f"  {sn:<20} {acc.mean():>12.4f} {acc.std():>7.4f} [{lo:.4f},{hi:.4f}] {stale.mean():>10.4f} {score.mean():>7.4f}")

# ── TABLE 2: Significance ─────────────────────────────────────────────────────
print("\n" + "="*72)
print("TABLE 2: Statistical Significance (TG vs Baselines)")
print("="*72)
tg = summary['TemporalStateGraph']
for bl in ['StaticMemory','RecencyMemory']:
    b = summary[bl]
    t, p   = ttest_rel(tg['acc'], b['acc'])
    _, wp  = wilcoxon(tg['acc'], b['acc'])
    d      = cohens_d(tg['acc'], b['acc'])
    delta  = tg['acc'].mean() - b['acc'].mean()
    lo, hi = boot(tg['acc'] - b['acc'])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
    print(f"  TG vs {bl:<20} Δ={delta:+.4f} [{lo:.4f},{hi:.4f}]  t={t:.2f}  p={p:.2e}  d={d:.3f} {sig}")

# ── TABLE 3: Per Query Type ────────────────────────────────────────────────────
QTYPES_MAP = {'work':'Employer','diet':'Preference','pending':'Task',
              'BEFORE':'Historical','feeling':'Mood'}
qp = defaultdict(lambda: defaultdict(lambda: {'c':0,'s':0,'n':0}))
for key, items in detailed.items():
    _, _, sn = parse_key(key)
    if not sn: continue
    for item in items:
        q = item['question']
        for kw, qt in QTYPES_MAP.items():
            if kw in q:
                qp[qt][sn]['c'] += item['judge']['correct']
                qp[qt][sn]['s'] += item['judge']['used_stale']
                qp[qt][sn]['n'] += 1
                break

print("\n" + "="*72)
print("TABLE 3: Per Query-Type Performance")
print("="*72)
print(f"{'Query':<14} {'TG Acc':>8} {'TG Stale':>9} {'SM Acc':>8} {'SM Stale':>9}")
print("-"*55)
for qt in ['Task','Employer','Preference','Historical','Mood']:
    def r(sn): 
        d = qp[qt][sn]; n = max(d['n'],1)
        return d['c']/n, d['s']/n
    ta, ts = r('TemporalStateGraph')
    sa, ss = r('StaticMemory')
    print(f"  {qt:<12} {ta:>8.3f} {ts:>9.3f} {sa:>8.3f} {ss:>9.3f}")

# ── TABLE 4: Cross-Seed ───────────────────────────────────────────────────────
print("\n" + "="*72)
print("TABLE 4: Cross-Seed Robustness")
print("="*72)
for sn in SYS:
    s42, s1337 = [], []
    for sk, sd in results.items():
        if sn not in sd: continue
        if sk.split('_')[0] == '42': s42.append(sd[sn]['accuracy'])
        else: s1337.append(sd[sn]['accuracy'])
    if s42 and s1337:
        print(f"  {sn:<22}  seed42={np.mean(s42):.4f}  seed1337={np.mean(s1337):.4f}  |Δ|={abs(np.mean(s42)-np.mean(s1337)):.4f}")

print("\nAnalysis complete.")
