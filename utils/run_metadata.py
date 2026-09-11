import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path


def _git(repo_dir: Path, *args: str, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        capture_output=True,
        text=text,
        timeout=10,
    ).stdout


def _json_safe(value):
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def capture_git_state(repo_dir):
    """Return Git provenance and a binary-safe patch of tracked changes."""
    repo_dir = Path(repo_dir).resolve()
    try:
        root = Path(_git(repo_dir, "rev-parse", "--show-toplevel").strip())
        commit = _git(root, "rev-parse", "HEAD").strip()
        branch = _git(root, "branch", "--show-current").strip() or None
        status_lines = _git(
            root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        patch = _git(root, "diff", "HEAD", "--binary", "--no-ext-diff", text=False)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "error": str(exc),
        }, b""

    untracked_files = [line[3:] for line in status_lines if line.startswith("?? ")]
    return {
        "available": True,
        "repository_root": str(root),
        "commit": commit,
        "short_commit": commit[:12],
        "branch": branch,
        "dirty": bool(status_lines),
        "status": status_lines,
        "untracked_files": untracked_files,
        "diff_sha256": hashlib.sha256(patch).hexdigest() if patch else None,
        "patch_file": "git_diff.patch" if patch else None,
    }, patch


def create_run_metadata(repo_dir, config, command=None):
    """Capture the code state and JSON-safe training configuration."""
    git_state, patch = capture_git_state(repo_dir)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": list(sys.argv if command is None else command),
        "git": git_state,
        "config": _json_safe(config),
    }
    return metadata, patch


def save_run_metadata(run_dir, metadata, patch):
    """Write provenance files and return the paths that were created."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if patch:
        patch_path = run_dir / metadata["git"]["patch_file"]
        patch_path.write_bytes(patch)
        paths.append(patch_path)

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return [metadata_path, *paths]
