import json
import os
from datetime import datetime
from collections import defaultdict

# Gather all data
data = []
base_path = "/home/videep/aria_data"

for glasses_id in os.listdir(base_path):
    glasses_path = os.path.join(base_path, glasses_id)
    if not os.path.isdir(glasses_path):
        continue
    
    for f in os.listdir(glasses_path):
        if f.endswith('.vrs.json'):
            json_path = os.path.join(glasses_path, f)
            try:
                with open(json_path) as jf:
                    d = json.load(jf)
                    vrs_file = f.replace('.json', '')
                    data.append({
                        'glasses_id': glasses_id,
                        'filename': vrs_file,
                        'start_time': d.get('start_time'),
                        'file_size': d.get('file_size'),
                        'path': os.path.join(glasses_path, vrs_file)
                    })
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

# Sort by start_time
data.sort(key=lambda x: x['start_time'] if x['start_time'] else 0)

# Print sorted data with human readable time
print("=" * 120)
print(f"{'Glasses':<8} {'Start Time':<22} {'Size (GB)':<12} {'Filename'}")
print("=" * 120)

for d in data:
    ts = d['start_time']
    dt = datetime.fromtimestamp(ts) if ts else "N/A"
    size_gb = d['file_size'] / (1024**3) if d['file_size'] else 0
    print(f"{d['glasses_id']:<8} {str(dt):<22} {size_gb:>10.2f}   {d['filename']}")

# Now group by approximate timestamps (within 10 minutes of each other = same game)
print("\n" + "=" * 120)
print("GROUPING BY GAME (recordings within ~10 min of each other)")
print("=" * 120)

games = []
current_game = []
last_time = None

for d in data:
    ts = d['start_time']
    if last_time is None or (ts - last_time) < 600:  # within 10 minutes
        current_game.append(d)
    else:
        if current_game:
            games.append(current_game)
        current_game = [d]
    last_time = ts

if current_game:
    games.append(current_game)

for i, game in enumerate(games, 1):
    print(f"\n--- GAME {i} ---")
    print(f"Time range: {datetime.fromtimestamp(game[0]['start_time'])} - {datetime.fromtimestamp(game[-1]['start_time'])}")
    game_sizes = [g['file_size'] / (1024**3) for g in game]
    print(f"Glasses involved: {sorted(set(g['glasses_id'] for g in game))}")
    print(f"File sizes: {[f'{s:.2f} GB' for s in game_sizes]}")
    for g in game:
        dt = datetime.fromtimestamp(g['start_time'])
        size_gb = g['file_size'] / (1024**3)
        print(f"  {g['glasses_id']}: {g['filename']} ({size_gb:.2f} GB) @ {dt}")

# Generate the proposed directory structure
print("\n" + "=" * 120)
print("PROPOSED NEW DIRECTORY STRUCTURE")
print("=" * 120)

for i, game in enumerate(games, 1):
    print(f"\ngame{i}/")
    for g in sorted(game, key=lambda x: x['glasses_id']):
        print(f"  {g['glasses_id']}/")
        print(f"    {g['filename']}")
        print(f"    {g['filename']}.json")
