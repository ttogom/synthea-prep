"""Preserve stopped experiment artifacts without changing their existing contents."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
from zoneinfo import ZoneInfo
from dotenv import dotenv_values

ROOT=Path('/Users/rasikaramanan/Documents/usc/by_semester/f26/csci566/project/synthea-prep')
STAMP='20261112'
ARCHIVE=ROOT/'experiments/output/speedup_handoff_artifacts_20261112'
BACKUP=Path('/Users/rasikaramanan/Documents/medcopilot-backups/speedup-20261112')
PILOT=ROOT/'experiments/output/2026-11-11_232045PST__pilot_speedup_so__speedup'
START=datetime(2026,11,11,tzinfo=ZoneInfo('America/Los_Angeles')).timestamp()
EXCLUDED_NAMES={'.env','.run.lock','.initialization.lock','.DS_Store'}
EXCLUDED_PARTS={'__pycache__','.git'}

def is_excluded(p):
    return p.name in EXCLUDED_NAMES or p.name.startswith('.env.') or p.suffix=='.pyc' or bool(set(p.parts)&EXCLUDED_PARTS) or p.name.endswith('.lock')

def walk(base):
    if base.is_symlink() or base.is_file():
        if not is_excluded(base): yield base
        return
    for directory, dirs, files in os.walk(base,followlinks=False):
        parent=Path(directory)
        for name in list(dirs):
            p=parent/name
            if is_excluded(p): dirs.remove(name)
            elif p.is_symlink():
                dirs.remove(name); yield p
        for name in files:
            p=parent/name
            if not is_excluded(p): yield p

def sha(p):
    h=hashlib.sha256()
    if p.is_symlink(): h.update(os.readlink(p).encode());return h.hexdigest()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def copy_exact(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if source.is_symlink():
        target=os.readlink(source)
        if dest.is_symlink() and os.readlink(dest)==target:return
        if dest.exists() or dest.is_symlink():raise RuntimeError('Destination conflict: '+str(dest))
        dest.symlink_to(target)
    elif not dest.exists() or sha(source)!=sha(dest): shutil.copy2(source,dest)
    if sha(source)!=sha(dest):raise RuntimeError('Copy verification failed: '+str(source))

def scan_secrets(paths):
    secrets=[]
    for env in (ROOT/'.env',ROOT/'med_copilot/data/guidelines/.env'):
        if env.exists():
            for k,v in dotenv_values(env).items():
                if v and len(v)>12 and ('KEY' in k.upper() or 'TOKEN' in k.upper()) and '${' not in v:
                    secrets.append(v.encode())
    for k,v in os.environ.items():
        if ('API_KEY' in k or 'HF_TOKEN' in k) and v and len(v)>12:secrets.append(v.encode())
    findings=[]
    for p in paths:
        if p.is_symlink():continue
        tail=b''
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):
                data=tail+block
                if any(s in data for s in secrets):findings.append(str(p));break
                tail=data[-max([len(s) for s in secrets]or[1]):]
    if findings:
        print(json.dumps({'secret_bearing_paths':findings}));raise RuntimeError('Secret-bearing artifacts require sanitized archival copies; nothing committed')

def main():
    ARCHIVE.mkdir(parents=True,exist_ok=True);BACKUP.mkdir(parents=True,exist_ok=True)
    os.chmod(BACKUP,0o700)
    lock=PILOT/'.run.lock'
    with lock.open('r') as held:
        fcntl.flock(held.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        items={}
        known=[ROOT/'experiments/configs/pilot_speedup.yaml',ROOT/'experiments/input/pilot_speedup_so.jsonl',ROOT/'experiments/input/pilot_speedup_so.selection.json',PILOT]
        known.extend(ROOT.glob('experiments/output/ollama-calibration-speedup*'))
        known.extend(ROOT.glob('experiments/output/*speedup*validation*.json'))
        known.extend(ROOT.glob('experiments/output/*recovery*validation*.json'))
        for base in known:
            if base.exists() or base.is_symlink():
                for p in walk(base):items[str(p)]=(p,p.relative_to(ROOT),'pilot_in_place')
        for parent,label in ((Path('/private/tmp'),'private_tmp'),(Path('/var/folders/f2/td3s0h7d4d3ckqf_d6xgwm7h0000gn/T'),'session_tmp')):
            for base in parent.glob('medcopilot*'):
                if is_excluded(base):continue
                for p in walk(base):
                    relative=p.relative_to(parent)
                    destination=ARCHIVE/'external'/label/relative
                    items[str(p)]=(p,destination.relative_to(ROOT),'external_pilot_or_validation')
        # Existing cache predates all this work; only current query diagnostics qualify.
        for folder in (ROOT/'med_copilot/data/guidelines/cache',ROOT/'med_copilot/data/guidelines/logs'):
            for p in walk(folder):
                if not p.is_symlink() and p.stat().st_mtime>=START:
                    dest=ARCHIVE/'project_diagnostics'/p.relative_to(ROOT)
                    items[str(p)]=(p,dest.relative_to(ROOT),'current_project_query_cache_or_log')
        scan_secrets([x[0] for x in items.values()]);print('Secret scan passed for',len(items),'source artifacts',flush=True)
        records=[]
        for source,relative,category in sorted(items.values(),key=lambda x:str(x[1])):
            original_sha=sha(source)
            dest=ROOT/relative
            if source!=dest:copy_exact(source,dest)
            target=BACKUP/'repository'/relative
            copy_exact(source,target)
            record={'repo_path':str(relative),'source_path':str(source),'backup_path':str(target),'category':category,'kind':'symlink' if source.is_symlink() else 'file','sha256':original_sha,'size_bytes':len(os.readlink(source).encode()) if source.is_symlink() else source.stat().st_size,'source_mtime_ns':source.lstat().st_mtime_ns}
            if source.is_symlink():record['symlink_target']=os.readlink(source)
            if sha(source)!=original_sha:raise RuntimeError('Source changed during backup: '+str(source))
            records.append(record)
        print('Byte-verified backup complete:',len(records),'artifacts',flush=True)
        # Preserve original available configurations without changing any pilot files.
        snapshots=ARCHIVE/'reference_snapshots';snapshots.mkdir(exist_ok=True)
        content=subprocess.check_output(['git','show','2596e317a8197a267fca00dca7a94de8ce2d767e:med_copilot/data/guidelines/settings.yaml'],cwd=ROOT)
        original=snapshots/'guideline_settings_pre_rate_tuning.yaml'
        if original.exists() and original.read_bytes()!=content:raise RuntimeError('Original snapshot conflict')
        original.write_bytes(content)
        for source,name in ((ROOT/'experiments/configs/pilot_speedup.yaml','pilot_speedup.yaml'),(ROOT/'experiments/configs/condition_plan_models.yaml','condition_plan_models.yaml'),(ROOT/'med_copilot/data/guidelines/settings.yaml','guideline_settings_after_rate_tuning.yaml')):
            target=snapshots/name
            if not target.exists():shutil.copy2(source,target)
        sqlite_info={}
        database=PILOT/'jobs.sqlite3'
        if database.exists():
            source=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
            try:
                assert source.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                consistent=snapshots/'jobs.consistent.sqlite3'
                if not consistent.exists():
                    target=sqlite3.connect(consistent)
                    try:source.backup(target)
                    finally:target.close()
                check=sqlite3.connect(consistent.as_uri()+'?mode=ro',uri=True)
                try:
                    assert check.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                    tables=[r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
                    rows={table:source.execute('SELECT COUNT(*) FROM "'+table+'"').fetchone()[0] for table in tables}
                    assert rows=={table:check.execute('SELECT COUNT(*) FROM "'+table+'"').fetchone()[0] for table in tables}
                    sqlite_info={'integrity':'ok','tables_row_counts':rows,'consistent_snapshot':str(consistent.relative_to(ROOT)),'source_wal_size':(database.parent/'jobs.sqlite3-wal').stat().st_size if (database.parent/'jobs.sqlite3-wal').exists() else None}
                finally:check.close()
            finally:source.close()
        for p in walk(snapshots):
            relative=p.relative_to(ROOT);target=BACKUP/'repository'/relative;copy_exact(p,target)
            records.append({'repo_path':str(relative),'source_path':str(p),'backup_path':str(target),'category':'available_reference_snapshot_or_consistent_database','kind':'file','sha256':sha(p),'size_bytes':p.stat().st_size})
        historical=[]
        for line in subprocess.check_output(['git','ls-tree','-r','main','--','experiments/output'],cwd=ROOT,text=True).splitlines():
            meta,path=line.split('\t',1);mode,kind,blob=meta.split();historical.append({'path':path,'mode':mode,'type':kind,'blob':blob})
        tracked_resources=subprocess.check_output(['git','ls-tree','-r','--name-only','2596e317a8197a267fca00dca7a94de8ce2d767e','--','med_copilot/data','experiments/input/test_so.jsonl','experiments/input/test_labels.jsonl'],cwd=ROOT,text=True).splitlines()
        cache_files=list((ROOT/'med_copilot/data/guidelines/cache').rglob('*'));cache_files=[p for p in cache_files if p.is_file()]
        manifest_path=ARCHIVE/'artifact_manifest.json'
        manifest={'schema_version':1,'created_at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),'repo_root':str(ROOT),'backup_root':str(BACKUP),'artifact_root':str(ARCHIVE.relative_to(ROOT)),'base_commit':'2596e317a8197a267fca00dca7a94de8ce2d767e','artifacts':records,'manifest_repo_path':str(manifest_path.relative_to(ROOT)),'artifact_repo_paths':sorted({r['repo_path'] for r in records}|{str(manifest_path.relative_to(ROOT))}),'historical_main_outputs':historical,'production_resources_already_tracked':tracked_resources,'reusable_utilities_production':[{'original_path':'scripts/benchmark_ollama.py','final_production_path':'scripts/experiment_runtime/benchmark_ollama.py'},{'original_path':'scripts/med_copilot_pilot.py','final_production_path':'scripts/experiment_runtime/pilot.py'}],'exclusions':[{'scope':'credentials, .env files, coordination *.lock files, __pycache__ and bytecode','reason':'Not persistent scientific artifacts; credentials/coordination locks must not be committed'},{'scope':'med_copilot/data/guidelines/cache','reason':'All pre-existing GraphRAG cache files are dated 2025-10-01; none created by current work','files':len(cache_files),'size_bytes':sum(p.stat().st_size for p in cache_files)},{'scope':'med_copilot/data/guidelines/.cache; user HuggingFace/Ollama/NLTK caches','reason':'Pre-existing production download/runtime dependencies; not one-off pilot outputs'},{'scope':'symlink targets to med_copilot/upstream','reason':'Referenced production code is already retained in base Git commit; literal symlink targets preserved'}],'sqlite':sqlite_info,'secret_scan':{'status':'passed','matching_secret_values_written':False},'verification':{'all_sources_copied_byte_identical':True,'all_backups_sha256_verified':True,'original_pilot_files_not_rewritten':True,'manifest_self_hash_excluded':True}}
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        copy_exact(manifest_path,BACKUP/'artifact_manifest.json')
        paths_file=Path('/private/tmp/speedup_artifact_paths_20261112.nul')
        paths_file.write_bytes(b''.join(path.encode()+b'\0' for path in manifest['artifact_repo_paths']))
        print(json.dumps({'manifest':str(manifest_path),'backup':str(BACKUP),'artifacts':len(records),'bytes':sum(r['size_bytes'] for r in records),'sqlite':sqlite_info,'historical_outputs':len(historical),'maximum_file_bytes':max(r['size_bytes'] for r in records)},indent=2),flush=True)
        fcntl.flock(held.fileno(),fcntl.LOCK_UN)

if __name__=='__main__':main()
