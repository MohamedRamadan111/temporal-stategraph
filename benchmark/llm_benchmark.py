"""
llm_benchmark.py — Main benchmark runner.

Run:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python benchmark/llm_benchmark.py

Uses JSON cache to avoid rerunning completed LLM calls.
Results saved to results/final_results.json and results/detailed_outputs.json.
"""
from __future__ import annotations
import os, json, time, hashlib, traceback
from collections import defaultdict
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from temporal_stategraph import TemporalStateGraph, StaticMemory, RecencyMemory, SlidingWindow
from benchmark.scenarios import build_scenario

# ── Config ───────────────────────────────────────────────────────────────────
SEEDS           = [42, 1337]
NUM_SCENARIOS   = 50
MODEL           = "openai/gpt-4o-mini"
RATE_LIMIT      = 1.1
CACHE_FILE      = "results/llm_cache.json"
RESULTS_FILE    = "results/final_results.json"
DETAILS_FILE    = "results/detailed_outputs.json"
PROGRESS_FILE   = "results/progress.json"
OPENROUTER_KEY  = os.environ.get("OPENROUTER_API_KEY", "")

Path("results").mkdir(exist_ok=True)

# ── Cache & persistence ───────────────────────────────────────────────────────
def _load(path, default):
    return json.load(open(path)) if Path(path).exists() else default

CACHE    = _load(CACHE_FILE,    {})
PROGRESS = _load(PROGRESS_FILE, {})
RESULTS  = _load(RESULTS_FILE,  {})
DETAILS  = _load(DETAILS_FILE,  {})

def _save_all():
    json.dump(CACHE,    open(CACHE_FILE,    "w"), indent=2)
    json.dump(PROGRESS, open(PROGRESS_FILE, "w"), indent=2)
    json.dump(RESULTS,  open(RESULTS_FILE,  "w"), indent=2)
    json.dump(DETAILS,  open(DETAILS_FILE,  "w"), indent=2)

# ── LLM call ─────────────────────────────────────────────────────────────────
def llm_call(prompt: str, system: str = "", max_tokens: int = 120) -> str:
    key = hashlib.md5((prompt + system + MODEL).encode()).hexdigest()
    if key in CACHE:
        return CACHE[key]
    import urllib.request
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": prompt}],
        "temperature": 0.0, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}",
                 "Content-Type": "application/json"}, method="POST")
    time.sleep(RATE_LIMIT)
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read())
    out = data["choices"][0]["message"]["content"].strip()
    CACHE[key] = out
    return out

# ── Judge ─────────────────────────────────────────────────────────────────────
def judge(question, answer, expected, stale):
    prompt = f"""You are evaluating temporal memory reasoning.

IMPORTANT RULE: If the assistant says "no information", "memory does not indicate",
or "no current [state] information" for a transient state (mood, feeling) that
should have decayed — THIS IS CORRECT BEHAVIOR. Mark correct=true, used_stale=false.

Question: {question}
Answer: {answer}
Expected (current): {expected}
Stale trap: {stale}

Return ONLY JSON: {{"correct":bool,"used_stale":bool,"score":1-5,"reason":"brief"}}"""
    raw = llm_call(prompt, max_tokens=100)
    try:
        return json.loads(raw)
    except:
        return {"correct": False, "used_stale": True, "score": 1, "reason": "parse_error"}

# ── Evaluate one system on one scenario ──────────────────────────────────────
def evaluate(memory, scenario):
    current_time = 0.0
    for hour, key, value, typ, imp in scenario["events"]:
        dt = hour - current_time; current_time = hour
        memory.advance_time(dt)
        memory.add_fact(key, value, typ, imp)

    results_row, details_row = {}, []
    correct = stale = score_sum = 0

    for q in scenario["queries"]:
        memory.advance_time(24)
        ctx = memory.get_context()
        prompt = f"""You are a temporal reasoning assistant.
Use ONLY the memory below. Answer briefly.

MEMORY:
{ctx}

QUESTION:
{q['question']}"""
        answer  = llm_call(prompt, max_tokens=80)
        judged  = judge(q["question"], answer, q["expected"], q["stale"])
        details_row.append({"question": q["question"], "answer": answer,
                             "judge": judged, "context": ctx})
        if judged["correct"]: correct += 1
        if judged["used_stale"]: stale += 1
        score_sum += judged["score"]

    n = len(scenario["queries"])
    return {
        "correct": correct, "stale": stale,
        "accuracy": correct/n, "stale_rate": stale/n,
        "avg_score": score_sum/n, "score": score_sum,
    }, details_row

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    systems = {
        "TemporalStateGraph": TemporalStateGraph,
        "StaticMemory":       StaticMemory,
        "RecencyMemory":      RecencyMemory,
        "SlidingWindow":      SlidingWindow,
    }

    print("="*60)
    print("TemporalStateGraph Benchmark")
    print(f"Model: {MODEL} | Scenarios: {len(SEEDS)*NUM_SCENARIOS}")
    print("="*60)

    for seed in SEEDS:
        for idx in range(NUM_SCENARIOS):
            sk = f"{seed}_{idx}"
            if PROGRESS.get(sk) == "done":
                print(f"  skip {sk}")
                continue

            scenario = build_scenario(seed, idx)
            row_results, row_details = {}, {}

            for sys_name, SysCls in systems.items():
                det_key = f"{seed}_{idx}_{sys_name}"
                if det_key in DETAILS:
                    continue
                mem = SysCls()
                try:
                    metrics, details = evaluate(mem, scenario)
                    row_results[sys_name] = metrics
                    row_details[det_key]  = details
                    print(f"  {sys_name:<24} {sk}  "
                          f"acc={metrics['accuracy']:.2f}  stale={metrics['stale_rate']:.2f}")
                except Exception as e:
                    print(f"  ERROR {sys_name} {sk}: {e}")
                    traceback.print_exc()

            RESULTS[sk] = row_results
            DETAILS.update(row_details)
            PROGRESS[sk] = "done"
            _save_all()

    print("\nDone. Results in results/")

if __name__ == "__main__":
    main()
