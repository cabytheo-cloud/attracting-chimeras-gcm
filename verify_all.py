"""
Runs the three exact verifications (standard library only, about 30 s in total):
  three_units/verify_exact.py   main example: Propositions 3.1, 3.3, 3.4 and Theorem 3.2
  four_units/verify_exact.py    four-unit example: Theorem 3.6
  four_piece/verify_exact.py    four-piece example: Theorem 6.1
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent
failed = []
for folder in ("three_units", "four_units", "four_piece"):
    print(f"\n######## {folder}/verify_exact.py\n", flush=True)
    r = subprocess.run([sys.executable, "verify_exact.py"], cwd=ROOT / folder)
    out_ok = r.returncode == 0
    if not out_ok: failed.append(folder)
print("\n" + ("All exact verifications passed." if not failed else "Failures in: " + ", ".join(failed)))
sys.exit(1 if failed else 0)
