import json, os, glob
from datetime import datetime
from collections import defaultdict
from pathlib import Path

base = str(Path(__file__).resolve().parents[2] / "data")
games = sorted([d for d in os.listdir(base) if d.startswith("Game") and os.path.isdir(os.path.join(base, d))],
               key=lambda x: int(x[4:]))

def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}

# Collect per game: glass -> list of recordings (main vs tests)
game_data = {}
for g in games:
    gp = os.path.join(base, g)
    main = defaultdict(list)   # glass label -> recs
    tests = defaultdict(list)
    for jp in glob.glob(os.path.join(gp, "**", "*.vrs.json"), recursive=True):
        rel = os.path.relpath(jp, gp).split(os.sep)
        d = load(jp)
        rec = {
            "path": jp,
            "start": d.get("start_time"),
            "end": d.get("end_time"),
            "size": d.get("file_size"),
            "device": d.get("device_id"),
            "file": d.get("filename"),
        }
        if rel[0] == "tests":
            tests[rel[1]].append(rec)
        else:
            main[rel[0]].append(rec)
    game_data[g] = (main, tests)

def fmt(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"

print("="*110)
print("PER-GAME OVERVIEW (main glasses only)")
print("="*110)

summary = []
for g in games:
    main, tests = game_data[g]
    glasses = sorted(main.keys())
    # window = min start, max end across all main recs
    all_starts = [r["start"] for gl in main.values() for r in gl if r["start"]]
    all_ends = [r["end"] for gl in main.values() for r in gl if r["end"]]
    gmin, gmax = (min(all_starts) if all_starts else None), (max(all_ends) if all_ends else None)
    summary.append((g, len(glasses), gmin, gmax))
    print(f"\n### {g}  | {len(glasses)} glasses: {glasses}" + (f"  | +tests:{sorted(tests.keys())}" if tests else ""))
    if gmin:
        span = (gmax - gmin)/60.0
        print(f"    union window: {fmt(gmin)} -> {fmt(gmax)}  (span {span:.1f} min)")
    # per glass
    for gl in glasses:
        for r in main[gl]:
            dur = (r["end"]-r["start"])/60.0 if r["start"] and r["end"] else 0
            sz = r["size"]/1e9 if r["size"] else 0
            print(f"      [{gl:>4}] {fmt(r['start'])} -> {fmt(r['end'])}  {dur:5.1f}min  {sz:6.2f}GB  dev={r['device']}")

print("\n\n" + "="*110)
print("ALIGNMENT CHECK  (does each glass overlap the game's common window?)")
print("="*110)
# Common overlap window = max of starts .. min of ends (per glass take the rec; if multiple recs use union per glass)
for g in games:
    main, tests = game_data[g]
    if not main:
        continue
    # per-glass union span
    glass_span = {}
    for gl, recs in main.items():
        ss = [r["start"] for r in recs if r["start"]]
        ee = [r["end"] for r in recs if r["end"]]
        if ss and ee:
            glass_span[gl] = (min(ss), max(ee))
    if len(glass_span) < 2:
        print(f"\n{g}: only {len(glass_span)} glass with timing — cannot cross-check")
        continue
    latest_start = max(s for s,e in glass_span.values())
    earliest_end = min(e for s,e in glass_span.values())
    overlap = earliest_end - latest_start
    status = "ALIGNED" if overlap > 0 else "NO COMMON OVERLAP"
    print(f"\n{g}: common overlap = {overlap/60.0:6.1f} min  -> {status}")
    # flag glasses that barely overlap or are outliers
    for gl,(s,e) in sorted(glass_span.items()):
        # pairwise overlap with the consensus median window
        starts = sorted(x[0] for x in glass_span.values())
        ends = sorted(x[1] for x in glass_span.values())
        med_s = starts[len(starts)//2]; med_e = ends[len(ends)//2]
        ov = (min(e,med_e)-max(s,med_s))
        tag = ""
        if ov <= 0:
            tag = "  <<< NO OVERLAP WITH GROUP — LIKELY MISPLACED"
        elif abs(s-med_s) > 1800 or abs(e-med_e) > 1800:
            tag = "  <-- starts/ends >30min off from group"
        print(f"      [{gl:>4}] {fmt(s)} -> {fmt(e)}{tag}")

print("\n\n" + "="*110)
print("SUMMARY: glass counts")
print("="*110)
low = [(g,n) for g,n,_,_ in summary if n < 6]
for g,n,gmin,gmax in summary:
    flag = "  <-- LESS THAN 6" if n < 6 else ""
    print(f"  {g:<8} {n} glasses{flag}")
print(f"\nGames with <6 glasses: {[g for g,n in low]}")

# Cross-game device sanity: same device appearing in overlapping time in two games?
print("\n\n" + "="*110)
print("CROSS-GAME CHECK: same device_id with overlapping times across different games")
print("="*110)
dev_recs = defaultdict(list)
for g in games:
    main, _ = game_data[g]
    for gl, recs in main.items():
        for r in recs:
            if r["device"] and r["start"]:
                dev_recs[r["device"]].append((g, gl, r["start"], r["end"]))
for dev, lst in dev_recs.items():
    lst.sort(key=lambda x: x[2])
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            g1,gl1,s1,e1 = lst[i]; g2,gl2,s2,e2 = lst[j]
            if g1!=g2 and e1 and s2 and s2 < e1:  # overlap
                print(f"  device {dev[:8]} overlaps: {g1}/{gl1} ({fmt(s1)}-{fmt(e1)}) && {g2}/{gl2} ({fmt(s2)}-{fmt(e2)})")
