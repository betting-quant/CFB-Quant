"""Use --check first; --apply makes a backup and installs verified source."""
import argparse
import hashlib
from pathlib import Path
import ast
import shutil

TARGET=Path('src/cfb_quant/features/matchup_engine.py')
REPLACEMENT=Path('CFB-Quant-Kickoff-Chronology-Fix-v3/src/cfb_quant/features/matchup_engine.py')
# Supports extraction into CFB-Quant root or a nested folder with this script.
BUNDLE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser()
 g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--check',action='store_true')
 g.add_argument('--apply',action='store_true')
 a=p.parse_args()
 assert TARGET.exists(), f'Run from CFB-Quant root; missing {TARGET}'
 target_bytes=TARGET.read_bytes()
 expected=(BUNDLE/'expected_original_sha256.txt').read_text().strip()
 if hashlib.sha256(target_bytes).hexdigest()!=expected:
  raise RuntimeError('Your matchup_engine.py differs from the audited version. STOP; no changes made. Upload the current file before patching.')
 fixed=(BUNDLE/'src/cfb_quant/features/matchup_engine.py').read_text(encoding='utf-8')
 ast.parse(fixed)
 if a.check:
  print('PASS: original source matches, replacement compiles, no files changed')
  return
 backup=TARGET.with_name('matchup_engine.py.before_kickoff_fix.bak')
 if backup.exists():
  raise FileExistsError(f'Backup already exists: {backup}; no changes made')
 shutil.copy2(TARGET,backup)
 TARGET.write_text(fixed,encoding='utf-8')
 print('INSTALLED:',TARGET,'\nBACKUP:',backup)

if __name__=='__main__':
 main()
