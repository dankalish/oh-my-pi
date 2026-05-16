from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from robomp import host_tools
from robomp.db import Database
from robomp.github_backend import GitHubBackend
from robomp.github_client import IssueInfo, RepoInfo
from robomp.sandbox import LocalGitTransport, SandboxManager, Workspace

pytestmark = pytest.mark.skipif(
    os.environ.get("ROBOMP_PERMISSION_E2E") != "1",
    reason="set ROBOMP_PERMISSION_E2E=1 to run slot-permission e2e tests",
)

_SLOT_ONE = 2001
_SLOT_TWO = 2002
_SHARED_OMP_GID = 2000
_AUTHOR_NAME = "robomp-bot"
_AUTHOR_EMAIL = "robomp-bot@example.invalid"
_REPO = "octo/permission-e2e"


def _require_linux_root_toolchain() -> None:
    if platform.system() != "Linux" or os.geteuid() != 0:
        pytest.skip("slot permission e2e tests require Linux root so subprocesses can drop to omp-N UIDs")
    missing = [cmd for cmd in ("git", "bun", "cargo", "python3") if shutil.which(cmd) is None]
    if missing:
        pytest.skip(f"slot permission e2e tests require tools on PATH: {', '.join(missing)}")


def _git(args: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _write_seed_repo(seed: Path) -> None:
    (seed / "src").mkdir(parents=True)
    (seed / "crates" / "core" / "src").mkdir(parents=True)
    (seed / "package.json").write_text(
        json.dumps(
            {
                "name": "permission-e2e",
                "private": True,
                "type": "module",
                "scripts": {
                    "check": "bun run check:ts && cargo check --workspace",
                    "check:ts": "biome check src/index.ts",
                    "fix": "biome check --write --unsafe src/index.ts",
                },
                "devDependencies": {"@biomejs/biome": "^2.4.14"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (seed / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (seed / "src" / "index.ts").write_text("export const answer = 42;\n", encoding="utf-8")
    (seed / "Cargo.toml").write_text(
        '[workspace]\nmembers = ["crates/core"]\nresolver = "2"\n',
        encoding="utf-8",
    )
    (seed / "rust-toolchain.toml").write_text(
        '[toolchain]\nchannel = "stable"\nprofile = "minimal"\n',
        encoding="utf-8",
    )
    (seed / "crates" / "core" / "Cargo.toml").write_text(
        '[package]\nname = "permission-e2e-core"\nversion = "0.1.0"\nedition = "2021"\n\n[lib]\npath = "src/lib.rs"\n',
        encoding="utf-8",
    )
    (seed / "crates" / "core" / "src" / "lib.rs").write_text(
        "pub fn answer() -> u32 {\n    42\n}\n",
        encoding="utf-8",
    )


@pytest.fixture
def slot_tmp_path() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="robomp-permission-e2e-", dir="/tmp"))
    root.chmod(0o755)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _share_tree_with_slots(path: Path) -> None:
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        os.chown(root_path, 0, _SHARED_OMP_GID)
        root_path.chmod(0o2770)
        for dirname in dirs:
            child = root_path / dirname
            os.chown(child, 0, _SHARED_OMP_GID)
            child.chmod(0o2770)
        for filename in files:
            child = root_path / filename
            executable = child.stat().st_mode & 0o111
            os.chown(child, 0, _SHARED_OMP_GID)
            child.chmod(0o770 if executable else 0o660)


@pytest.fixture
def upstream_repo(slot_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    upstream = slot_tmp_path / "upstream.git"
    seed = slot_tmp_path / "seed"
    seed.mkdir()
    _write_seed_repo(seed)

    _git(["init", "--initial-branch=main", "--bare", str(upstream)], cwd=slot_tmp_path)
    _git(["init", "--initial-branch=main", str(seed)], cwd=slot_tmp_path)
    _git(["-C", str(seed), "add", "."], cwd=slot_tmp_path)
    commit_env = os.environ | {
        "GIT_AUTHOR_NAME": "seed",
        "GIT_AUTHOR_EMAIL": "seed@example.invalid",
        "GIT_COMMITTER_NAME": "seed",
        "GIT_COMMITTER_EMAIL": "seed@example.invalid",
    }
    _git(["-C", str(seed), "commit", "-m", "seed"], cwd=slot_tmp_path, env=commit_env)
    _git(["-C", str(seed), "remote", "add", "origin", str(upstream)], cwd=slot_tmp_path)
    _git(["-C", str(seed), "push", "origin", "main"], cwd=slot_tmp_path)
    _share_tree_with_slots(upstream)
    git_system_config = slot_tmp_path / "git-system.conf"
    _git(["config", "--file", str(git_system_config), "--add", "safe.directory", str(upstream)], cwd=slot_tmp_path)
    git_system_config.chmod(0o644)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(git_system_config))
    return upstream


@pytest.fixture
def tool_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def _ensure_workspace(
    root: Path, upstream: Path, *, number: int, slot_uid: int, existing_branch: str | None = None
) -> Workspace:
    manager = SandboxManager(root, transport=LocalGitTransport(token=None))
    return manager.ensure_workspace(
        repo=_REPO,
        number=number,
        title="permission e2e",
        clone_url=str(upstream),
        default_branch="main",
        existing_branch=existing_branch,
        author_name=_AUTHOR_NAME,
        author_email=_AUTHOR_EMAIL,
        slot_uid=slot_uid,
    )


def _bindings(
    *,
    db: Database,
    tool_loop: asyncio.AbstractEventLoop,
    workspace: Workspace,
    upstream: Path,
    slot_uid: int,
) -> host_tools.ToolBindings:
    repo = RepoInfo(full_name=_REPO, default_branch="main", clone_url=str(upstream), private=False)
    issue = IssueInfo(
        repo=_REPO,
        number=workspace.issue_number,
        title="permission e2e",
        body="",
        state="open",
        author="human",
        labels=(),
        is_pull_request=False,
    )
    return host_tools.ToolBindings(
        db=db,
        github=cast(GitHubBackend, object()),  # not used by these local-only host-tool paths
        git_transport=LocalGitTransport(token=None),
        repo=repo,
        issue=issue,
        workspace=workspace,
        loop=tool_loop,
        author_name=_AUTHOR_NAME,
        author_email=_AUTHOR_EMAIL,
        slot_uid=slot_uid,
    )


def _run_ok(
    bindings: host_tools.ToolBindings,
    cmd: list[str] | tuple[str, ...],
    *,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[str]:
    proc = host_tools._run_repo_command(bindings, cmd, timeout=timeout)
    assert proc.returncode == 0, (
        f"command failed as slot {bindings.slot_uid}: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc


def _write_as_slot(bindings: host_tools.ToolBindings, relative_path: str, content: str) -> None:
    _run_ok(
        bindings,
        [
            "python3",
            "-c",
            (
                "from pathlib import Path; "
                "Path(__import__('sys').argv[1]).parent.mkdir(parents=True, exist_ok=True); "
                "Path(__import__('sys').argv[1]).write_text(__import__('sys').argv[2], encoding='utf-8')"
            ),
            relative_path,
            content,
        ],
    )


def _prepare_shared_cargo_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cargo_home = tmp_path / "shared-cache" / "cargo"
    cargo_target = tmp_path / "shared-cache" / "cargo-target"
    for path in (cargo_home, cargo_target):
        path.mkdir(parents=True)
        os.chown(path, 0, _SHARED_OMP_GID)
        path.chmod(0o2770)
    monkeypatch.setenv("CARGO_HOME", str(cargo_home))
    monkeypatch.setenv("CARGO_TARGET_DIR", str(cargo_target))
    return cargo_target


def test_slot_workspace_runs_bun_biome_cargo_and_git_after_root_reentry(
    slot_tmp_path: Path,
    upstream_repo: Path,
    db: Database,
    tool_loop: asyncio.AbstractEventLoop,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_linux_root_toolchain()
    cargo_target = _prepare_shared_cargo_cache(slot_tmp_path, monkeypatch)
    workspaces = slot_tmp_path / "workspaces"

    first = _ensure_workspace(workspaces, upstream_repo, number=101, slot_uid=_SLOT_ONE)
    stale_bun_cache = first.root / ".omp-xdg" / "cache" / "bun-install" / "root-owned-stale"
    stale_bun_cache.mkdir(parents=True, exist_ok=True)
    stale_marker = stale_bun_cache / "marker.txt"
    stale_marker.write_text("root-owned\n", encoding="utf-8")
    stale_bun_cache.chmod(0o700)
    stale_marker.chmod(0o600)

    workspace = _ensure_workspace(
        workspaces,
        upstream_repo,
        number=101,
        slot_uid=_SLOT_ONE,
        existing_branch=first.branch,
    )
    bindings = _bindings(db=db, tool_loop=tool_loop, workspace=workspace, upstream=upstream_repo, slot_uid=_SLOT_ONE)

    _run_ok(bindings, ["bun", "install", "--no-progress"], timeout=300.0)
    _run_ok(bindings, ["bun", "run", "check:ts"], timeout=180.0)
    _run_ok(bindings, ["cargo", "check", "--workspace"], timeout=600.0)
    host_tools._run_pre_publish_bun_check(bindings, {}, tool_name="gh_push_branch", stage="push")

    runtime_env = host_tools._repo_command_env(bindings)
    bun_cache = Path(runtime_env["BUN_INSTALL_CACHE_DIR"])
    assert bun_cache.is_dir()
    assert bun_cache.stat().st_uid == _SLOT_ONE
    assert stale_marker.stat().st_uid == _SLOT_ONE
    assert (cargo_target / "debug").is_dir()
    assert (cargo_target / "debug").stat().st_gid == _SHARED_OMP_GID

    _write_as_slot(bindings, "src/slot-generated.ts", "export const generatedBySlot = true;\n")
    _run_ok(bindings, ["git", "add", "src/slot-generated.ts", "Cargo.lock", "bun.lock"])
    _run_ok(bindings, ["git", "commit", "-m", "slot generated file"])
    status = _run_ok(bindings, ["git", "status", "--porcelain", "--untracked-files=normal"])
    assert status.stdout.strip() == ""


def test_git_pool_metadata_survives_root_push_and_retry_slot(
    slot_tmp_path: Path,
    upstream_repo: Path,
    db: Database,
    tool_loop: asyncio.AbstractEventLoop,
) -> None:
    _require_linux_root_toolchain()
    workspaces = slot_tmp_path / "workspaces"

    first = _ensure_workspace(workspaces, upstream_repo, number=102, slot_uid=_SLOT_ONE)
    first_bindings = _bindings(db=db, tool_loop=tool_loop, workspace=first, upstream=upstream_repo, slot_uid=_SLOT_ONE)
    _write_as_slot(first_bindings, "src/first-slot.ts", "export const firstSlot = 1;\n")
    _run_ok(first_bindings, ["git", "add", "src/first-slot.ts"])
    _run_ok(first_bindings, ["git", "commit", "-m", "first slot commit"])

    first_head = host_tools._guarded_push_branch(first_bindings, {}, "gh_push_branch", first.branch)
    remote_head = _git(["--git-dir", str(upstream_repo), "rev-parse", first.branch], cwd=slot_tmp_path).stdout.strip()
    assert remote_head == first_head

    retry = _ensure_workspace(
        workspaces,
        upstream_repo,
        number=102,
        slot_uid=_SLOT_TWO,
        existing_branch=first.branch,
    )
    retry_bindings = _bindings(db=db, tool_loop=tool_loop, workspace=retry, upstream=upstream_repo, slot_uid=_SLOT_TWO)

    _run_ok(retry_bindings, ["git", "fsck", "--no-progress"], timeout=180.0)
    _write_as_slot(retry_bindings, "src/retry-slot.ts", "export const retrySlot = 2;\n")
    _run_ok(retry_bindings, ["git", "add", "src/retry-slot.ts"])
    _run_ok(retry_bindings, ["git", "commit", "-m", "retry slot commit"])

    retry_head = host_tools._guarded_push_branch(retry_bindings, {}, "gh_push_branch", retry.branch)
    remote_retry_head = _git(
        ["--git-dir", str(upstream_repo), "rev-parse", retry.branch], cwd=slot_tmp_path
    ).stdout.strip()
    assert remote_retry_head == retry_head
    assert retry_head != first_head
