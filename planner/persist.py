"""Everything that makes a Colab run survive a disconnect.

- `persist_dir`: a directory on Google Drive (fails loudly if Drive cannot be mounted).
- `StageCache`: finished stages (evals, data builds) are stored as JSON and skipped on re-run.
- `latest_checkpoint`: resume point for the HF/TRL trainers.
- `wandb_init`: resumes the same W&B run after a restart.
- `push_with_retry`: Hub uploads that tolerate flaky networking.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable

DRIVE_MOUNT = "/content/drive"
DRIVE_ROOT = "/content/drive/MyDrive/tool_use_demo"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def persist_dir(name: str, use_drive: bool = True, root: str = DRIVE_ROOT, local_root: str = "outputs") -> str:
    """Directory that outlives the runtime: Drive on Colab, `outputs/<name>` elsewhere.

    If Drive is requested but cannot be mounted this raises instead of quietly
    writing to the ephemeral disk, where a disconnect would lose everything.
    """
    if use_drive and in_colab():
        if not os.path.isdir(os.path.join(DRIVE_MOUNT, "MyDrive")):
            from google.colab import drive

            drive.mount(DRIVE_MOUNT)
        if not os.path.isdir(os.path.join(DRIVE_MOUNT, "MyDrive")):
            raise RuntimeError("Google Drive is not mounted. Mount it, or set USE_DRIVE = False to accept losing progress on disconnect.")
        path = os.path.join(root, name)
    else:
        path = os.path.join(local_root, name)
    os.makedirs(path, exist_ok=True)
    return path


def atomic_write_json(path: str, obj: Any) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class StageCache:
    """Results of finished stages, keyed by name, stored as JSON in the persistent directory.

    `check_config` fingerprints the run settings: if they changed since the
    cache was written, old results are ignored (and the changed keys printed),
    so a different experiment never silently reuses stale numbers.
    """

    def __init__(self, directory: str, enabled: bool = True):
        self.directory = os.path.join(directory, "stages")
        self.enabled = enabled
        os.makedirs(self.directory, exist_ok=True)

    def path(self, name: str) -> str:
        return os.path.join(self.directory, f"{name}.json")

    def has(self, name: str) -> bool:
        return self.enabled and os.path.exists(self.path(name))

    def load(self, name: str) -> Any:
        return read_json(self.path(name))

    def save(self, name: str, obj: Any) -> Any:
        atomic_write_json(self.path(name), obj)
        return obj

    def get_or_run(self, name: str, fn: Callable[[], Any]) -> Any:
        if self.has(name):
            print(f"[cache] {name}: reusing saved result")
            return self.load(name)
        return self.save(name, fn())

    def check_config(self, config: dict) -> bool:
        """Return True if the stored config matches; otherwise disable reuse for this session."""
        path = self.path("_config")
        config = json.loads(json.dumps(config, default=str))
        if os.path.exists(path):
            old = read_json(path)
            changed = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
            if changed:
                print(f"[cache] config changed ({', '.join(changed)}): cached stage results will not be reused")
                self.enabled = False
                return False
            return True
        atomic_write_json(path, config)
        return True

    def done(self) -> list[str]:
        if not self.enabled:
            return []
        return sorted(f[:-5] for f in os.listdir(self.directory) if f.endswith(".json") and not f.startswith("_"))


def latest_checkpoint(output_dir: str) -> str | None:
    """Newest complete `checkpoint-N` directory, or None."""
    if not os.path.isdir(output_dir):
        return None
    best, best_step = None, -1
    for name in os.listdir(output_dir):
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        path = os.path.join(output_dir, name)
        # trainer_state.json is written last, so its presence marks a complete checkpoint.
        if match and os.path.exists(os.path.join(path, "trainer_state.json")) and int(match.group(1)) > best_step:
            best, best_step = path, int(match.group(1))
    return best


def read_secrets(names=("HF_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN")) -> dict[str, bool]:
    """Copy Colab Secrets into environment variables; report which ones are available."""
    found = {}
    userdata = None
    if in_colab():
        from google.colab import userdata  # type: ignore
    for name in names:
        if not os.environ.get(name) and userdata is not None:
            try:
                value = userdata.get(name)
            except Exception:  # noqa: BLE001 - secret missing or access not granted
                value = None
            if value:
                os.environ[name] = value
        found[name] = bool(os.environ.get(name))
    return found


def wandb_init(directory: str, project: str | None, run_name: str, config: dict | None = None, entity: str | None = None):
    """Start (or resume) a W&B run. Returns (run or None, report_to string for trainers).

    The run id is stored next to the checkpoints, so after a disconnect the
    same run continues and the curves stay in one piece.
    """
    if not project or not os.environ.get("WANDB_API_KEY"):
        print("W&B disabled (set WANDB_API_KEY and a project to enable it)")
        os.environ["WANDB_MODE"] = "disabled"
        return None, "none"
    import wandb

    id_path = os.path.join(directory, "wandb_run_id.txt")
    if os.path.exists(id_path):
        with open(id_path) as f:
            run_id = f.read().strip()
    else:
        run_id = wandb.util.generate_id()
        with open(id_path, "w") as f:
            f.write(run_id)
    os.environ["WANDB_PROJECT"] = project
    run = wandb.init(project=project, entity=entity, name=run_name, id=run_id, resume="allow", config=config or {})
    return run, "wandb"


def push_with_retry(fn: Callable[[], Any], what: str, retries: int = 3, wait: float = 20.0) -> bool:
    """Run an upload, retrying on failure. Returns False (after printing) instead of raising."""
    for attempt in range(1, retries + 1):
        try:
            fn()
            print(f"[hub] pushed {what}")
            return True
        except Exception as exc:  # noqa: BLE001 - network / hub errors
            print(f"[hub] pushing {what} failed (attempt {attempt}/{retries}): {exc}")
            if attempt < retries:
                time.sleep(wait * attempt)
    return False


def model_card(title: str, base_model: str, description: str, sections: dict[str, str]) -> str:
    body = [
        "---",
        f"base_model: {base_model}",
        "library_name: transformers",
        "tags: [tool-calling, function-calling, planner, unsloth, trl]",
        "---",
        f"# {title}",
        "",
        description,
    ]
    for heading, text in sections.items():
        body += ["", f"## {heading}", "", text]
    return "\n".join(body) + "\n"


def log_summary(run, prefix: str, values: dict) -> None:
    """Write flat numeric results (e.g. a benchmark summary) to the W&B run summary."""
    if run is None:
        return
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            run.summary[f"{prefix}/{key}"] = value
