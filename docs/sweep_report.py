"""
Aggregate slurm/logs/sweep/result_*.tsv (written by slurm/sweep_budget_array*.slurm)
into one table per benchmark and pick a recommended config.

Selection rule ("收益最大、时间又差不多"): among configs whose wall time is at
most `--time-factor` x the baseline (s=2, t=2, i=1000, no backtrack) AND whose
collar check equals the baseline's expected count, take the smallest volume;
ties -> fewer seeds, then smaller t, then smaller i, then no backtrack.

Usage: sweep_report.py [--time-factor 1.5] [--dir slurm/logs/sweep]
"""

import argparse
import collections
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "slurm", "logs", "sweep"))
ap.add_argument("--time-factor", type=float, default=1.5)
ap.add_argument("--json", default=None, help="write the picks as JSON {bench: {s,t,i,bt,vol,wall,base_vol,base_wall}}")
args = ap.parse_args()

rows = []
for fn in glob.glob(os.path.join(args.dir, "result_*.tsv")):
    with open(fn) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            bench, s, t, it, vol, chk, wall, fb = p[:8]
            bt = 1 if (len(p) > 8 and p[8] == "bt") else 0
            try:
                vol = float(vol)
            except ValueError:
                vol = None
            rows.append(dict(bench=bench, s=int(s), t=int(t), i=int(it), vol=vol, chk=chk,
                             wall=int(wall), fb=int(fb) if fb.isdigit() else 0, bt=bt))

by = collections.defaultdict(list)
for r in rows:
    by[r["bench"]].append(r)

order = ["bv_16", "dj_16", "ghz_16", "vqe_16", "wstate_16", "qaoa_16", "grover_6", "qft_16", "qpe_16"]
picks = {}
for bench in order:
    rs = by.get(bench)
    if not rs:
        continue
    base = [r for r in rs if (r["s"], r["t"], r["i"], r["bt"]) == (2, 2, 1000, 0)]
    base = base[0] if base else None
    print(f"\n=== {bench} ===  baseline: " + (f"vol={base['vol']} wall={base['wall']}s collars={base['chk']}" if base else "(not in yet)"))
    print(f"  {'s':>2} {'t':>2} {'i':>5} {'bt':>2} | {'volume':>8} {'d_vol':>7} | {'wall':>5} {'x':>4} | {'collars':>8} fb")
    for r in sorted(rs, key=lambda r: (r["bt"], r["i"], r["t"], r["s"])):
        dv = f"{(r['vol'] - base['vol']) / base['vol'] * 100:+.1f}%" if (base and r["vol"] is not None and base["vol"]) else ""
        xw = f"{r['wall'] / base['wall']:.2f}" if (base and base["wall"]) else ""
        print(f"  {r['s']:>2} {r['t']:>2} {r['i']:>5} {r['bt']:>2} | {str(r['vol']):>8} {dv:>7} | {r['wall']:>5} {xw:>4} | {r['chk']:>8} {r['fb']}")
    if base and base["vol"] is not None:
        ok = [r for r in rs if r["vol"] is not None and r["chk"] == base["chk"] and r["wall"] <= args.time_factor * base["wall"]]
        best = min(ok, key=lambda r: (r["vol"], r["s"], r["t"], r["i"], r["bt"])) if ok else base
        picks[bench] = (best, base)
        tag = "" if best is not base else "  (no better config within the time budget)"
        print(f"  -> pick: s={best['s']} t={best['t']} i={best['i']} bt={best['bt']}  vol {base['vol']} -> {best['vol']}  wall {base['wall']}s -> {best['wall']}s{tag}")

print("\n=== summary (time factor <= %.2f) ===" % args.time_factor)
print(f"{'bench':<10} {'baseline':>9} {'pick':>9} {'d_vol':>7} {'wall':>12}  config")
for bench in order:
    if bench in picks:
        b, base = picks[bench]
        dv = (b["vol"] - base["vol"]) / base["vol"] * 100
        print(f"{bench:<10} {base['vol']:>9} {b['vol']:>9} {dv:>+6.1f}% {str(base['wall'])+'->'+str(b['wall'])+'s':>12}  -s {b['s']} -t {b['t']} -i {b['i']}" + ("  --backtrack 1" if b["bt"] else ""))

if args.json:
    import json
    out = {b: dict(s=p_[0]["s"], t=p_[0]["t"], i=p_[0]["i"], bt=p_[0]["bt"], vol=p_[0]["vol"], wall=p_[0]["wall"],
                   collars=p_[0]["chk"], base_vol=p_[1]["vol"], base_wall=p_[1]["wall"]) for b, p_ in picks.items()}
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"picks written to {args.json}")
