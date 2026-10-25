"""Re-run the sealed diagnosis with every attempt to open a *truth* file raising an error.

Run from the repo root:  python docs/evidence/blocked_truth_diagnosis.py outputs/diag_blocked
Shows the committed verdict does not depend on data/sealed_v1_truth.json (output: .log next to this file)."""

import builtins, io, json, pathlib, sys

OUT = pathlib.Path(sys.argv[1])
_open, _io_open = builtins.open, io.open


def guard(p):
    if "truth" in str(p).lower():
        raise PermissionError(f"BLOCKED truth access: {p}")


def g_open(f, *a, **k):
    guard(f)
    return _open(f, *a, **k)


builtins.open = g_open
io.open = g_open
for name in ("read_text", "read_bytes", "open"):
    orig = getattr(pathlib.Path, name)

    def wrap(self, *a, _o=orig, **k):
        guard(self)
        return _o(self, *a, **k)

    setattr(pathlib.Path, name, wrap)

for probe in (
    lambda: open("data/sealed_v1_truth.json"),
    lambda: pathlib.Path("data/sealed_v1_truth.json").read_text(),
):
    try:
        probe()
        sys.exit("guard failed: truth file was readable")
    except PermissionError:
        pass
print("guard active: truth file unreadable")

from heatinv.diagnose import diagnose_bundle  # noqa: E402

diagnose_bundle("data/sealed_v1", "configs/diagnosis_rules.yaml", OUT)
new = json.loads(_open(OUT / "diagnosis.json").read())
old = json.loads(_open("results/sealed_diagnosis.json").read())
print("verdict new/old:", new["verdict"], "/", old["verdict"])
print("top new/old:", new["top_hypothesis"], "/", old["top_hypothesis"])
same = [
    abs(a["delta_chi2"] - b["delta_chi2"])
    for a, b in zip(new["hypothesis_tests"], old["hypothesis_tests"])
]
print("max |d delta_chi2| across hypotheses:", max(same))
print(
    "data sha256 same:",
    new["provenance"]["data_sha256"] == old["provenance"]["data_sha256"],
)
