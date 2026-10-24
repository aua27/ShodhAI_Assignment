"""Provenance logging and checkpoints.

Every result file carries `provenance`: data version (bundle sha256), full config + its hash,
seed, git commit (+ dirty flag), hardware, thread count, library versions and wall time.

Checkpoints carry a fingerprint (config hash + data hash + schema). Resuming refuses any
checkpoint whose fingerprint differs, and skips files that fail to load (truncated writes).
Writes are atomic (temp file + rename) so an interruption cannot leave a half-written "latest".
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

CKPT_SCHEMA = 1


class CheckpointMismatch(RuntimeError):
    pass


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _git():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "src", "configs"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }
    proc = platform.processor()
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    proc = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info["cpu"] = proc
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    import numpy
    import scipy

    info["numpy"], info["scipy"] = numpy.__version__, scipy.__version__
    return info


def provenance(
    kind: str, cfg: dict, data_sha: str | None, seed=None, t_start: float | None = None
) -> dict:
    return {
        "kind": kind,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256": data_sha,
        "config_hash": config_hash(cfg),
        "config": cfg,
        "seed": seed,
        "git": _git(),
        "hardware": hardware(),
        "wall_time_s": None
        if t_start is None
        else round(time.perf_counter() - t_start, 3),
    }


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable))
    os.replace(tmp, path)
    return path


def _jsonable(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


class Checkpointer:
    def __init__(self, directory: Path, fingerprint: dict, keep: int = 3):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = {"schema": CKPT_SCHEMA, **fingerprint}
        self.keep = keep
        self.rejections: list[dict] = []

    def save(self, state: dict, step: int) -> Path:
        import torch

        path = self.dir / f"ckpt_{step:07d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": self.fingerprint, "step": step, "state": state}, tmp)
        os.replace(tmp, path)
        for old in sorted(self.dir.glob("ckpt_*.pt"))[: -self.keep]:
            old.unlink()
        return path

    def load_latest(self):
        """Newest compatible checkpoint, or None. Incompatible/corrupt files are recorded and skipped."""
        import torch

        for path in sorted(self.dir.glob("ckpt_*.pt"), reverse=True):
            try:
                blob = torch.load(path, weights_only=False)
            except Exception as e:  # truncated / corrupt
                self.rejections.append(
                    {"file": path.name, "reason": f"unreadable: {type(e).__name__}"}
                )
                continue
            fp = blob.get("fingerprint", {})
            diff = sorted(
                k
                for k in set(fp) | set(self.fingerprint)
                if fp.get(k) != self.fingerprint.get(k)
            )
            if diff:
                self.rejections.append(
                    {"file": path.name, "reason": f"fingerprint mismatch on {diff}"}
                )
                continue
            return blob["state"]
        return None
