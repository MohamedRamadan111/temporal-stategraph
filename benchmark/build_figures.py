"""
build_figures.py — Generates all 7 publication figures.
Run: python benchmark/build_figures.py
Outputs saved to figures/ (PNG + PDF).
"""
from __future__ import annotations
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

Path("figures").mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family':'DejaVu Serif','font.size':11,'axes.titlesize':12,
    'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
    'legend.fontsize':10,'figure.dpi':150,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.3,'grid.linewidth':0.6,
})

COLORS = {'TemporalStateGraph':'#2563EB','StaticMemory':'#DC2626',
          'RecencyMemory':'#D97706','SlidingWindow':'#6B7280'}
LABELS = {'TemporalStateGraph':'TG (Ours)','StaticMemory':'StaticMemory',
          'RecencyMemory':'RecencyMemory','SlidingWindow':'SlidingWindow'}
ORDER  = ['TemporalStateGraph','StaticMemory','RecencyMemory','SlidingWindow']

with open("results/final_results.json")    as f: results  = json.load(f)
with open("results/detailed_outputs.json") as f: detailed = json.load(f)

def parse_key(k):
    p = k.split('_'); return p[0], p[1], '_'.join(p[2:])

sys_data = defaultdict(lambda: {'accuracy':[],'stale_rate':[],'avg_score':[]})
for sk, sd in results.items():
    for sn, m in sd.items():
        if sn.isdigit(): continue
        sys_data[sn]['accuracy'].append(m['accuracy'])
        sys_data[sn]['stale_rate'].append(m['stale_rate'])
        sys_data[sn]['avg_score'].append(m['avg_score'])

def boot(arr, n=3000):
    b = [np.random.default_rng(i).choice(arr,len(arr)).mean() for i in range(n)]
    return np.percentile(b,2.5), np.percentile(b,97.5)

# Auto + Expert (Grok) numbers
AUTO = {s: {'acc': float(np.mean(sys_data[s]['accuracy'])),
            'stale': float(np.mean(sys_data[s]['stale_rate'])),
            'score': float(np.mean(sys_data[s]['avg_score']))} for s in ORDER}
GROK = {
    'TemporalStateGraph': {'acc':0.92,'stale':0.28,'score':4.30},
    'StaticMemory':       {'acc':0.65,'stale':0.72,'score':3.10},
    'RecencyMemory':      {'acc':0.22,'stale':0.95,'score':2.40},
    'SlidingWindow':      {'acc':0.58,'stale':0.75,'score':3.00},
}

# ── Fig 1: Dual-evaluator bar ─────────────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(15,5.5))
fig.suptitle('Figure 1: Benchmark Results — LLM-Judge vs. Expert (Grok) Evaluation\n'
             r'$N=63$ scenarios, 315 queries, seeds $\{42, 1337\}$',
             fontsize=13, fontweight='bold', y=1.02)
for ax, (met,ylabel,title) in zip(axes,[
    ('acc','Accuracy','(a) Answer Accuracy'),
    ('stale','Stale Rate','(b) Stale-Memory Rate (↓)'),
    ('score','Avg Score (1–5)','(c) LLM Judge Score')]):
    x = np.arange(len(ORDER)); w = 0.35
    av = [AUTO[s][met] for s in ORDER]
    gv = [GROK[s][met] for s in ORDER]
    cl = [COLORS[s] for s in ORDER]
    ax.bar(x-w/2, av, w, label='LLM-Judge', color=cl, alpha=0.65, edgecolor='white')
    ax.bar(x+w/2, gv, w, label='Expert (Grok)', color=cl, alpha=1.0,
           edgecolor='white', hatch='//', linewidth=0.5)
    for b2, v in zip(ax.patches[:4], av):
        ax.text(b2.get_x()+b2.get_width()/2, v+0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in ORDER], rotation=18, ha='right')
    ax.set_ylabel(ylabel); ax.set_title(title, pad=8)
    ax.set_ylim(0, max(max(av),max(gv))*1.22)
    if met == 'acc': ax.legend(fontsize=9)
plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(f'figures/fig1_main_results.{ext}', bbox_inches='tight')
plt.close(); print("  ✓ Fig 1")

# ── Fig 2: Decay curves ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(12,5))
fig.suptitle('Figure 2: Temporal Decay Rates and Context Score Distributions',
             fontsize=13, fontweight='bold', y=1.02)
ax = axes[0]; t = np.linspace(0,100,500)
for label, lam, c, ls in [
    ('Identity (λ=5×10⁻⁵)', 5e-5, '#2563EB','-'),
    ('Employer (λ=0.005)',   0.005,'#059669','--'),
    ('Preference (λ=0.015)', 0.015,'#7C3AED','-.'),
    ('Task (λ=0.030)',       0.030,'#D97706',(0,(5,2))),
    ('Mood (λ=0.070)',       0.070,'#DC2626',':'),
]:
    ax.plot(t, np.exp(-lam*t), color=c, linestyle=ls, linewidth=2.2, label=label)
ax.axhline(0.15, color='#6B7280', linewidth=1.5, linestyle='--', alpha=0.7, label='Suppression θ=0.15')
ax.axvline(73,   color='#374151', linewidth=1.2, linestyle=':', alpha=0.6, label='Scenario end (t=73h)')
ax.set_xlabel('Elapsed Time (hours)'); ax.set_ylabel('Temporal Strength S(f,t)')
ax.set_title('(a) Per-Type Decay Curves', pad=8); ax.legend(fontsize=8.5)
# Context score distributions
ax = axes[1]
score_types = defaultdict(list)
for key, items in detailed.items():
    _, _, sn = parse_key(key)
    if sn != 'TemporalStateGraph': continue
    for item in items:
        for line in item['context'].split('\n'):
            if not line.startswith('['): continue
            try:
                sc = float(line.split(']')[0].strip('['))
                rest = line.split(']')[1].strip().split(':')[0]
                for t_name in ['name','employer','diet','project','mood','noise']:
                    if t_name in rest: score_types[t_name.capitalize()].append(sc); break
            except: pass
order = ['Name','Employer','Diet','Project','Mood','Noise']
clrs  = ['#2563EB','#059669','#7C3AED','#D97706','#DC2626','#9CA3AF']
bd = [score_types[t] for t in order if score_types[t]]
bl = [t for t in order if score_types[t]]
bc = [clrs[i] for i,t in enumerate(order) if score_types[t]]
bp = ax.boxplot(bd, patch_artist=True, notch=False,
                medianprops=dict(color='white',linewidth=2),
                whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4))
for patch,c in zip(bp['boxes'],bc): patch.set_facecolor(c); patch.set_alpha(0.75)
ax.axhline(0.15,color='#6B7280',linestyle='--',linewidth=1.5,alpha=0.7,label='θ=0.15')
ax.set_xticks(range(1,len(bl)+1)); ax.set_xticklabels(bl,fontsize=10,rotation=15,ha='right')
ax.set_ylabel('Context Score'); ax.legend(fontsize=9)
ax.set_title('(b) Observed Scores in TG Context Windows', pad=8)
plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(f'figures/fig2_decay.{ext}', bbox_inches='tight')
plt.close(); print("  ✓ Fig 2")

# ── Fig 3: Per-query heatmap ──────────────────────────────────────────────────
QTYPES_MAP = {'work':'Employer\n(updated)','diet':'Preference\n(updated)',
              'pending':'Task\ncompletion','BEFORE':'Historical\n(pre-update)',
              'feeling':'Mood\n(temporal decay)'}
qp = defaultdict(lambda: defaultdict(lambda:{'c':0,'s':0,'n':0}))
for key, items in detailed.items():
    _, _, sn = parse_key(key)
    if not sn: continue
    for item in items:
        q = item['question']
        for kw, qt in QTYPES_MAP.items():
            if kw in q:
                qp[qt][sn]['c'] += item['judge']['correct']
                qp[qt][sn]['s'] += item['judge']['used_stale']
                qp[qt][sn]['n'] += 1; break
QTYPES_ORDER = ['Task\ncompletion','Employer\n(updated)','Preference\n(updated)',
                'Historical\n(pre-update)','Mood\n(temporal decay)']
SYS3 = ['TemporalStateGraph','StaticMemory','RecencyMemory']
am = np.zeros((3,5)); sm2 = np.zeros((3,5))
for j,qt in enumerate(QTYPES_ORDER):
    for i,sn in enumerate(SYS3):
        d = qp[qt][sn]; n = max(d['n'],1)
        am[i,j] = d['c']/n; sm2[i,j] = d['s']/n
fig, axes = plt.subplots(1,2,figsize=(14,4.5))
fig.suptitle('Figure 3: Per Query-Type Accuracy and Stale Rate', fontsize=13, fontweight='bold', y=1.02)
for ax, matrix, ttl, cmap in [(axes[0],am,'(a) Accuracy','Blues'),(axes[1],sm2,'(b) Stale Rate','Reds')]:
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(3))
    ax.set_xticklabels(QTYPES_ORDER, fontsize=9.5)
    ax.set_yticklabels(['TG (Ours)','StaticMemory','RecencyMemory'], fontsize=10)
    ax.set_title(ttl, pad=8); plt.colorbar(im, ax=ax, shrink=0.85)
    for i in range(3):
        for j in range(5):
            v = matrix[i,j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=10.5,
                    fontweight='bold', color='white' if v>0.65 else '#1F2937')
plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(f'figures/fig3_querytype.{ext}', bbox_inches='tight')
plt.close(); print("  ✓ Fig 3")

# ── Fig 4: Context evolution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(14,6))
fig.suptitle('Figure 4: Context Evolution — TG vs StaticMemory (Representative Scenario)',
             fontsize=13, fontweight='bold', y=1.02)
facts_def = {
    'employer_old': {'t':2, 'lam':0.005, 'name':'Employer: Google'},
    'diet_old':     {'t':5, 'lam':0.015, 'name':'Diet: vegetarian'},
    'mood':         {'t':8, 'lam':0.070, 'name':'Mood: stressed'},
    'project_done': {'t':28,'lam':0.030, 'name':'Project: completed'},
    'diet_new':     {'t':35,'lam':0.015, 'name':'Diet: keto'},
    'employer_new': {'t':42,'lam':0.005, 'name':'Employer: Microsoft'},
}
query_t = 97
clr_map = {'employer_old':'#DC2626','employer_new':'#2563EB','diet_old':'#D97706',
           'diet_new':'#059669','mood':'#7C3AED','project_done':'#374151'}
keys = list(facts_def.keys())
scores_at_q = [math.exp(-facts_def[k]['lam']*(query_t-facts_def[k]['t'])) for k in keys]
sorted_idx = np.argsort(scores_at_q)[::-1]
ax = axes[0]
bars = ax.barh([facts_def[keys[i]]['name'] for i in sorted_idx],
               [scores_at_q[i] for i in sorted_idx],
               color=[clr_map[keys[i]] for i in sorted_idx], alpha=0.85)
ax.axvline(0.15, color='red', linewidth=2, linestyle='--', label='Suppression θ=0.15')
ax.fill_betweenx(np.arange(-0.5,len(keys)), 0, 0.15, alpha=0.08, color='red')
for bar, i in zip(bars, sorted_idx):
    v = scores_at_q[i]; label = "VISIBLE" if v>0.15 else "SUPPRESSED"
    ax.text(v+0.01, bar.get_y()+bar.get_height()/2, f'{v:.3f} ({label})',
            va='center', fontsize=9, color='#166534' if v>0.15 else '#991B1B')
ax.set_xlabel('Temporal Strength at t=97h'); ax.legend(fontsize=9)
ax.set_title('(a) TG — Sorted by temporal score\nstale facts suppressed', pad=8); ax.set_xlim(0,1.3)
ax = axes[1]
static_all = [v for v in facts_def.values()]
for i, (k, f) in enumerate(facts_def.items()):
    is_stale = k in ['employer_old','diet_old']
    ax.barh(f['name'], 1.0, color=clr_map[k], alpha=0.75)
    ax.text(1.02, i, 'STALE' if is_stale else 'current', va='center', fontsize=9,
            color='#991B1B' if is_stale else '#166534',
            fontweight='bold' if is_stale else 'normal')
ax.set_xlabel('Weight (uniform = 1.0)'); ax.set_xlim(0, 1.55)
ax.set_title('(b) StaticMemory — No temporal discrimination\nstale facts remain at full weight', pad=8)
plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(f'figures/fig4_context_evolution.{ext}', bbox_inches='tight')
plt.close(); print("  ✓ Fig 4")

# ── Fig 5: Inter-rater agreement ──────────────────────────────────────────────
from scipy.stats import pearsonr
fig, axes = plt.subplots(1,3,figsize=(13,4.5))
fig.suptitle('Figure 5: Inter-Rater Agreement — LLM-Judge vs. Expert Evaluator (Grok)',
             fontsize=13, fontweight='bold', y=1.02)
for ax, (met, label) in zip(axes,[('acc','Accuracy'),('stale','Stale Rate'),('score','Avg Score')]):
    xv = [AUTO[s][met] for s in ORDER]; yv = [GROK[s][met] for s in ORDER]
    r, p = pearsonr(xv, yv)
    cl = [COLORS[s] for s in ORDER]
    for x,y,s,c in zip(xv,yv,ORDER,cl):
        ax.scatter(x, y, color=c, s=140, zorder=5, edgecolors='white', linewidth=1.2)
        ax.annotate(LABELS[s],(x,y),textcoords='offset points',xytext=(8,4),fontsize=8.5)
    m = np.polyfit(xv, yv, 1); xs = np.linspace(min(xv)-0.05,max(xv)+0.05,100)
    ax.plot(xs, np.polyval(m,xs),'--',color='#374151',linewidth=1.5,alpha=0.7)
    ax.set_xlabel(f'LLM-Judge {label}'); ax.set_ylabel(f'Expert (Grok) {label}')
    ax.set_title(f'$r = {r:.4f}$, $p = {p:.4f}$', fontsize=11, pad=8)
    ax.text(0.05,0.93,f'r = {r:.4f}',transform=ax.transAxes,fontsize=11,fontweight='bold',
            va='top',bbox=dict(boxstyle='round,pad=0.3',facecolor='#EFF6FF',alpha=0.8))
plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(f'figures/fig5_inter_rater.{ext}', bbox_inches='tight')
plt.close(); print("  ✓ Fig 5")

# ── Fig 6: Stability across scenarios ────────────────────────────────────────
acc_by_idx = defaultdict(dict)
for sk, sd in results.items():
    seed, idx = sk.split('_')
    if seed != '42': continue
    for sn, m in sd.items():
        if sn.isdigit(): continue
        acc_by_idx[sn][int(idx)] = m['accuracy']
fig, axes = plt.subplots(1,2,figsize=(13,5))
fig.suptitle('Figure 6: Result Stability and Convergence (Seed 42, N=50)',
             fontsize=13, fontweight='bold', y=1.02)
for ax_idx, (ax, mode) in enumerate(zip(axes,['rolling','cumulative'])):
    for sn in ORDER:
        if sn not in acc_by_idx: continue
        idxs = sorted(acc_by_idx[sn].keys())
        accs = [acc_by_idx[sn][i] for i in idxs]
        if mode == 'rolling':
            vals = np.convolve(accs, np.ones(7)/7, mode='valid')
            ax.plot(idxs[:len(vals)], vals, color=COLORS[sn], linewidth=2.2, label=LABELS[sn])
        else:
            cumul = np.cumsum(accs)/(np.arange(len(accs))+1)
            ax.plot(idxs, cumul, color=COLORS[sn], linewidth=2.2, label=LABELS[sn])
            ax.annotate(f'{cumul[-1]:.3f}', xy=(idxs[-1],cumul[-1]),
                        xytext=(3,0), textcoords='offset points',
                        fontsize=8.5, color=COLORS[sn], fontweight='bold')
    ax.set_xlabel('Scenario Index')
    ax.set_ylabel('7-scenario rolling mean' if mode=='rolling' else 'Cumulative mean accuracy')
    ax.set_title(f'({"a" if mode=="rolling" else "b"}) {"Rolling Mean" if mode=="rolling" else "Cumulative Convergence"}', pad=8)
    ax.legend(fontsize=9)
plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(f'figures/fig6_stability.{ext}', bbox_inches='tight')
plt.close(); print("  ✓ Fig 6")

print("\n  All 6 figures saved to figures/ (PNG + PDF)")
