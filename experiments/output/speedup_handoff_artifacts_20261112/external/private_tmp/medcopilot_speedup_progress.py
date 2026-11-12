import json
from pathlib import Path
import sqlite3

run = Path('/Users/rasikaramanan/Documents/usc/by_semester/f26/csci566/project/synthea-prep/experiments/output/2026-11-11_232045PST__pilot_speedup_so__speedup')
metadata = json.loads((run / 'run.json').read_text())
print({key: metadata.get(key) for key in ('status', 'active_generation_model', 'finished_at')})
print('errors:', len(metadata.get('errors', [])))
connection = sqlite3.connect((run / 'jobs.sqlite3').as_uri() + '?mode=ro', uri=True)
print(connection.execute("select stage,model,status,count(*) from jobs where stage != 'timing' group by stage,model,status").fetchall())
connection.close()
if metadata.get('plan_evaluation'):
    print('plan_evaluation:', {key: metadata['plan_evaluation'].get(key) for key in
          ('status', 'started_at', 'finished_at', 'concurrency', 'context_length', 'calls', 'cache_hits', 'counts')})
cache = run / 'judge_responses.jsonl'
if cache.exists():
    print('durable judge replies:', sum(bool(line.strip()) for line in cache.read_text().split('\n')))
