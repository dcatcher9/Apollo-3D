#!/usr/bin/env python3
"""Plan, run, resume, inspect, and verify Apollo's ordinal label corpus.

This is the unattended front door for the existing fail-closed orchestrator.
It deliberately owns no rendering, scoring, or publication semantics.  A
``plan`` freezes the exact request and immutable orchestration plan; ``run``
requires the printed plan digest and delegates checkpoint/resume to the
orchestrator; ``status`` is read-only; and ``verify`` revalidates every output.
Training remains a separate, explicit command after the corpus is complete.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess

import orchestrate_artistic_ordinal_labels as ordinal


SCHEMA = 4
CONTRACT = "apollo-artistic-ordinal-regeneration-request-v4"
REQUEST_FILENAME = "regeneration_request.json"
PLAN_FILENAME = "ordinal_orchestration_plan.json"
STATE_FILENAME = "ordinal_orchestration_state.json"
RUNTIME_FILENAME = "ordinal_orchestration_runtime.json"
LOCK_FILENAME = "ordinal_regeneration.lock"
LOCK_OWNER_FILENAME = "ordinal_regeneration_lock_owner.json"
RUNTIME_SCHEMA = 1
RUNTIME_CONTRACT = "apollo-ordinal-regeneration-runtime-v1"
LOCK_OWNER_SCHEMA = 1
LOCK_OWNER_CONTRACT = "apollo-ordinal-regeneration-lock-owner-v1"

DEFAULT_WORKSPACE = Path(
    r"E:\ApolloDev\artistic-policy\ordinal-v2-target-only-v9"
)
DEFAULT_ACTIVE_SPLIT = Path(
    r"E:\ApolloDev\artistic-policy\public-mono-hdr-bootstrap-v3\datasets"
    r"\active_artistic_split_sdr_native_pq_full_cadence.json"
)
DEFAULT_PYTHON = Path(
    r"E:\ApolloDev\venvs\artistic-policy\Scripts\python.exe"
)
DEFAULT_DEPTH_STATE_CACHE_ROOT = ordinal.DEFAULT_DEPTH_STATE_CACHE_ROOT
DEFAULT_SCORED_RESULT_CACHE_ROOT = ordinal.DEFAULT_SCORED_RESULT_CACHE_ROOT


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _lock_byte(stream):
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)


def _acquire_lock(stream):
    _lock_byte(stream)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_lock(stream):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _load_optional_object(path, description):
    path = Path(path)
    if not path.is_file():
        return None
    return ordinal.load_json(path, description)


@contextmanager
def _exclusive_run_lock(workspace, plan_sha256):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    lock_path = workspace / LOCK_FILENAME
    owner_path = workspace / LOCK_OWNER_FILENAME
    stream = lock_path.open("a+b")
    try:
        try:
            _acquire_lock(stream)
        except OSError as error:
            owner = _load_optional_object(
                owner_path, "ordinal regeneration lock owner"
            )
            detail = f"; owner={owner}" if owner else ""
            raise RuntimeError(
                f"ordinal regeneration is already running{detail}"
            ) from error
        owner = {
            "schema": LOCK_OWNER_SCHEMA,
            "contract": LOCK_OWNER_CONTRACT,
            "pid": os.getpid(),
            "host": platform.node(),
            "started_at": _utc_now(),
            "plan_sha256": plan_sha256,
        }
        ordinal.write_json_atomic(owner_path, owner)
        try:
            yield owner
        finally:
            owner_path.unlink(missing_ok=True)
            _release_lock(stream)
    finally:
        stream.close()


def _run_lock_active(workspace):
    path = Path(workspace) / LOCK_FILENAME
    if not path.is_file():
        return False
    with path.open("r+b") as stream:
        try:
            _acquire_lock(stream)
        except OSError:
            return True
        _release_lock(stream)
    return False


def _write_runtime(workspace, plan_sha256, state, **updates):
    path = Path(workspace) / RUNTIME_FILENAME
    existing = _load_optional_object(path, "ordinal regeneration runtime")
    if existing is not None and (
            existing.get("schema") != RUNTIME_SCHEMA or
            existing.get("contract") != RUNTIME_CONTRACT or
            existing.get("plan_sha256") != plan_sha256):
        existing = None
    value = dict(existing or {})
    value.update({
        "schema": RUNTIME_SCHEMA,
        "contract": RUNTIME_CONTRACT,
        "plan_sha256": plan_sha256,
        "state": state,
        "last_update_at": _utc_now(),
    })
    value.update(updates)
    ordinal.write_json_atomic(path, value)
    return value


def _request_path(workspace):
    return Path(workspace) / REQUEST_FILENAME


def _plan_path(workspace):
    return Path(workspace) / PLAN_FILENAME


def _load_request(workspace):
    workspace = Path(workspace).resolve()
    path = _request_path(workspace)
    value = ordinal.load_json(path, "ordinal regeneration request")
    if (value.get("schema") != SCHEMA or
            value.get("contract") != CONTRACT):
        raise RuntimeError(f"unsupported regeneration request: {path}")
    required = (
        "workspace", "active_split", "build_dir", "conf", "python",
        "run_prefix", "production", "clip_limit", "score_workers",
        "depth_state_cache_root", "scored_result_cache_root",
        "plan_sha256",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise RuntimeError(
            "regeneration request is missing: " + ", ".join(missing)
        )
    path_keys = ("workspace", "active_split", "build_dir", "conf", "python")
    if any(not isinstance(value[key], str) or not value[key]
           for key in path_keys):
        raise RuntimeError("regeneration request contains an invalid path")
    if not isinstance(value["run_prefix"], str) or not value["run_prefix"]:
        raise RuntimeError("regeneration request run_prefix is invalid")
    cache_root = value["depth_state_cache_root"]
    if cache_root is not None and (
            not isinstance(cache_root, str) or not cache_root):
        raise RuntimeError("regeneration request depth-state cache root is invalid")
    score_cache_root = value["scored_result_cache_root"]
    if score_cache_root is not None and (
            not isinstance(score_cache_root, str) or not score_cache_root):
        raise RuntimeError(
            "regeneration request scored-result cache root is invalid"
        )
    if Path(value["workspace"]).resolve() != workspace:
        raise RuntimeError(
            "regeneration request workspace does not match its location"
        )
    if (not isinstance(value["production"], list) or
            not all(isinstance(item, str) and item
                    for item in value["production"])):
        raise RuntimeError("regeneration request production list is invalid")
    if (not isinstance(value["score_workers"], int) or
            isinstance(value["score_workers"], bool) or
            value["score_workers"] < 1):
        raise RuntimeError("regeneration request score_workers is invalid")
    clip_limit = value["clip_limit"]
    if (clip_limit is not None and
            (not isinstance(clip_limit, int) or
             isinstance(clip_limit, bool) or clip_limit < 1)):
        raise RuntimeError("regeneration request clip_limit is invalid")
    digest = value["plan_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(character not in "0123456789abcdef" for character in digest)):
        raise RuntimeError("regeneration request plan_sha256 is invalid")
    return value


def _namespace(request, *, stop_after="catalog", restart=False):
    return argparse.Namespace(
        workspace=Path(request["workspace"]),
        active_split=Path(request["active_split"]),
        build_dir=Path(request["build_dir"]),
        conf=Path(request["conf"]),
        python=Path(request["python"]),
        run_prefix=request["run_prefix"],
        production=list(request["production"]),
        clip_limit=request["clip_limit"],
        score_workers=request["score_workers"],
        depth_state_cache_root=(
            Path(request["depth_state_cache_root"])
            if request["depth_state_cache_root"] is not None else None
        ),
        scored_result_cache_root=(
            Path(request["scored_result_cache_root"])
            if request["scored_result_cache_root"] is not None else None
        ),
        render_workers=1,
        stop_after=stop_after,
        dry_run=False,
        restart=restart,
    )


def _request_value(args, plan):
    plan_value = plan.as_dict()
    return {
        "schema": SCHEMA,
        "contract": CONTRACT,
        "workspace": str(plan.workspace),
        "active_split": str(plan.active_split),
        "build_dir": str(plan.build_dir),
        "conf": str(plan.conf),
        "python": str(plan.python),
        "run_prefix": args.run_prefix,
        "production": list(args.production or ()),
        "clip_limit": args.clip_limit,
        "score_workers": args.score_workers,
        "depth_state_cache_root": (
            str(plan.depth_state_cache_root)
            if plan.depth_state_cache_root is not None else None
        ),
        "scored_result_cache_root": (
            str(plan.scored_result_cache_root)
            if plan.scored_result_cache_root is not None else None
        ),
        "plan_sha256": ordinal.canonical_sha256(plan_value),
        "plan_contract": plan_value["contract"],
        "scope": plan_value["scope"],
        "training_eligible": plan_value["training_eligible"],
        "estimates": plan_value["estimates"],
    }


def _freeze(path, value, description):
    path = Path(path)
    if path.is_file():
        if ordinal.load_json(path, description) != value:
            raise RuntimeError(
                f"{description} differs; use a new workspace: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ordinal.write_json_atomic(path, value)


def _rebuild_plan(request, *, require_build=True):
    args = _namespace(request)
    if require_build:
        ordinal.run_eval.require_current_build(str(args.build_dir.resolve()))
    plan = ordinal.build_plan(args)
    plan_value = plan.as_dict()
    digest = ordinal.canonical_sha256(plan_value)
    if digest != request.get("plan_sha256"):
        raise RuntimeError(
            "regeneration inputs/code/build changed; create a new workspace"
        )
    frozen_plan = ordinal.load_json(
        _plan_path(plan.workspace), "ordinal orchestration plan"
    )
    if frozen_plan != plan_value:
        raise RuntimeError("frozen orchestration plan differs from current plan")
    return plan


def command_plan(args):
    ordinal.run_eval.require_current_build(str(args.build_dir.resolve()))
    namespace = argparse.Namespace(
        workspace=args.workspace,
        active_split=args.active_split,
        build_dir=args.build_dir,
        conf=args.conf,
        python=args.python,
        run_prefix=args.run_prefix,
        production=args.production,
        clip_limit=args.clip_limit,
        score_workers=args.score_workers,
        depth_state_cache_root=(
            None if getattr(args, "no_depth_state_cache", False) else
            getattr(args, "depth_state_cache_root", DEFAULT_DEPTH_STATE_CACHE_ROOT)
        ),
        scored_result_cache_root=(
            None if getattr(args, "no_scored_result_cache", False) else
            getattr(
                args, "scored_result_cache_root",
                DEFAULT_SCORED_RESULT_CACHE_ROOT,
            )
        ),
        render_workers=1,
    )
    plan = ordinal.build_plan(namespace)
    plan_value = plan.as_dict()
    request = _request_value(args, plan)
    _freeze(_request_path(plan.workspace), request, "regeneration request")
    _freeze(_plan_path(plan.workspace), plan_value, "ordinal orchestration plan")
    return {
        "command": "plan",
        "workspace": str(plan.workspace),
        "plan_sha256": request["plan_sha256"],
        "scope": request["scope"],
        "training_eligible": request["training_eligible"],
        "estimates": request["estimates"],
        "next_command": (
            f'"{plan.python}" "{Path(__file__).resolve()}" run '
            f'--workspace "{plan.workspace}" --accept-plan-sha256 '
            f'{request["plan_sha256"]}'
        ),
    }


def command_run(args):
    request = _load_request(args.workspace)
    expected = request.get("plan_sha256")
    if args.accept_plan_sha256 != expected:
        raise RuntimeError(
            "--accept-plan-sha256 does not match the frozen reviewed plan"
        )
    with _exclusive_run_lock(args.workspace, expected) as owner:
        _write_runtime(
            args.workspace, expected, "starting",
            pid=owner["pid"], host=owner["host"],
            run_started_at=owner["started_at"], current_step=None,
            current_phase="plan-validation", last_failure=None,
        )
        try:
            plan = _rebuild_plan(request, require_build=True)
            namespace = _namespace(
                request, stop_after=args.stop_after,
                restart=args.repair_partials,
            )
            _write_runtime(
                args.workspace, expected, "running",
                current_phase="source-verification",
            )
            result = ordinal.execute(plan, namespace)
        except BaseException as error:
            _write_runtime(
                args.workspace, expected, "failed",
                last_failure=(
                    f"{type(error).__name__}: {error}"
                )[-2000:],
            )
            raise
        _write_runtime(
            args.workspace, expected, "completed",
            current_step=None, current_phase=None,
            finished_at=_utc_now(), last_failure=None,
        )
        return result


def _status_value(workspace):
    workspace = Path(workspace).resolve()
    request = _load_request(workspace)
    plan = ordinal.load_json(_plan_path(workspace), "ordinal orchestration plan")
    if ordinal.canonical_sha256(plan) != request["plan_sha256"]:
        raise RuntimeError("frozen orchestration plan digest is invalid")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("frozen orchestration plan has no steps")
    if not all(isinstance(step, dict) for step in steps):
        raise RuntimeError("frozen orchestration plan has invalid steps")
    state_path = workspace / STATE_FILENAME
    state = (
        ordinal.load_json(state_path, "ordinal orchestration state")
        if state_path.is_file() else {}
    )
    completed = state.get("completed", [])
    if (not isinstance(completed, list) or
            not all(isinstance(key, str) and key for key in completed)):
        raise RuntimeError("ordinal orchestration state is invalid")
    if state and (
            state.get("schema") != plan.get("schema") or
            state.get("contract") != plan.get("contract") or
            state.get("plan_sha256") != request["plan_sha256"]):
        raise RuntimeError("ordinal orchestration state is stale")
    keys = [step.get("key") for step in steps]
    if (any(not isinstance(key, str) or not key for key in keys) or
            len(keys) != len(set(keys))):
        raise RuntimeError("frozen orchestration plan has invalid step keys")
    if (len(completed) != len(set(completed)) or
            any(key not in keys for key in completed)):
        raise RuntimeError("ordinal orchestration state contains invalid steps")
    remaining = [key for key in keys if key not in set(completed)]
    logs = workspace / "orchestration_logs"
    recent_log = None
    if logs.is_dir():
        candidates = [path for path in logs.iterdir() if path.is_file()]
        if candidates:
            recent_log = max(candidates, key=lambda path: path.stat().st_mtime)
    total = len(steps)
    runtime = _load_optional_object(
        workspace / RUNTIME_FILENAME, "ordinal regeneration runtime"
    )
    if runtime is not None and (
            runtime.get("schema") != RUNTIME_SCHEMA or
            runtime.get("contract") != RUNTIME_CONTRACT or
            runtime.get("plan_sha256") != request["plan_sha256"]):
        raise RuntimeError("ordinal regeneration runtime is stale")
    lock_active = _run_lock_active(workspace)
    lock_owner = (
        _load_optional_object(
            workspace / LOCK_OWNER_FILENAME,
            "ordinal regeneration lock owner",
        ) if lock_active else None
    )
    if lock_active:
        execution_state = "running"
    elif runtime and runtime.get("state") in {"starting", "running"}:
        execution_state = "interrupted"
    elif runtime and runtime.get("state") in {"failed", "completed"}:
        execution_state = runtime["state"]
    elif not completed:
        execution_state = "not-started"
    elif len(completed) == total:
        execution_state = "completed"
    else:
        execution_state = "stopped"
    resume_command = subprocess.list2cmdline([
        request["python"], str(Path(__file__).resolve()), "run",
        "--workspace", str(workspace), "--accept-plan-sha256",
        request["plan_sha256"],
    ])
    percent = round(100.0 * len(completed) / total, 2)
    return {
        "command": "status",
        "workspace": str(workspace),
        "plan_sha256": request["plan_sha256"],
        "completed_steps": len(completed),
        "total_steps": total,
        "percent_complete": percent,
        "step_percent_complete": percent,
        "progress_basis": "completed-plan-steps-unweighted",
        "execution_state": execution_state,
        "lock_active": lock_active,
        "lock_owner": lock_owner,
        "current_step": runtime.get("current_step") if runtime else None,
        "current_phase": runtime.get("current_phase") if runtime else None,
        "run_started_at": runtime.get("run_started_at") if runtime else None,
        "last_update_at": runtime.get("last_update_at") if runtime else None,
        "last_failure": runtime.get("last_failure") if runtime else None,
        "last_completed": completed[-1] if completed else None,
        "next_step": remaining[0] if remaining else None,
        "remaining_steps": len(remaining),
        "resume_command": resume_command,
        "recent_log": str(recent_log) if recent_log else None,
        "catalog": (
            str(workspace / "ordinal_frame_label_catalog.json")
            if (workspace / "ordinal_frame_label_catalog.json").is_file()
            else None
        ),
    }


def command_status(args):
    value = _status_value(args.workspace)
    if args.verify_completed:
        request = _load_request(args.workspace)
        plan = _rebuild_plan(request, require_build=False)
        state_path = plan.workspace / STATE_FILENAME
        state = (
            ordinal.load_json(state_path, "ordinal orchestration state")
            if state_path.is_file() else {"completed": []}
        )
        by_key = {step.key: step for step in plan.steps}
        completed = state.get("completed", ())
        known_steps = [by_key[key] for key in completed if key in by_key]
        results = iter(_verify_step_results(
            known_steps, plan,
            getattr(args, "workers", None) or request["score_workers"],
        ))
        invalid = [
            key for key in completed
            if key not in by_key or not next(results)
        ]
        if invalid:
            raise RuntimeError(
                "completed regeneration outputs failed validation: " +
                ", ".join(invalid[:4])
            )
        value["verified_completed_steps"] = len(state.get("completed", ()))
    return value


def _verify_step_results(steps, plan, workers):
    """Validate immutable publications concurrently in frozen-plan order."""
    if (not isinstance(workers, int) or isinstance(workers, bool) or
            workers < 1):
        raise ValueError("verification workers must be a positive integer")
    steps = tuple(steps)
    if workers == 1 or len(steps) <= 1:
        return [ordinal._step_complete(step, plan) for step in steps]
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(steps)),
            thread_name_prefix="ordinal-verify",
            ) as executor:
        futures = [
            executor.submit(ordinal._step_complete, step, plan)
            for step in steps
        ]
        # Resolve in plan order. Worker completion timing therefore cannot
        # change the first invalid publication reported to the caller.
        return [future.result() for future in futures]


def command_verify(args):
    request = _load_request(args.workspace)
    plan = _rebuild_plan(request, require_build=False)
    ordinal._verify_sources(plan)
    workers = getattr(args, "workers", None) or request["score_workers"]
    if args.completed_only:
        state_path = plan.workspace / STATE_FILENAME
        state = (
            ordinal.load_json(state_path, "ordinal orchestration state")
            if state_path.is_file() else {"completed": []}
        )
        by_key = {step.key: step for step in plan.steps}
        completed = state.get("completed", ())
        known_steps = [by_key[key] for key in completed if key in by_key]
        results = iter(_verify_step_results(known_steps, plan, workers))
        invalid = [
            key for key in completed
            if key not in by_key or not next(results)
        ]
        if invalid:
            raise RuntimeError(
                "recorded completed outputs failed validation: " +
                ", ".join(invalid[:4])
            )
    else:
        results = _verify_step_results(plan.steps, plan, workers)
        invalid = [
            step.key for step, valid in zip(plan.steps, results) if not valid
        ]
        if invalid:
            raise RuntimeError(
                f"regeneration is incomplete/invalid ({len(invalid)} steps); "
                f"first: {invalid[0]}"
            )
    result = _status_value(plan.workspace)
    result.update({
        "command": "verify",
        "verified": True,
        "verification_scope": (
            "completed-only" if args.completed_only else "complete-corpus"
        ),
        "source_hash_verification": "full-content-per-invocation",
        "verification_workers": workers,
    })
    return result


def _common_plan_arguments(parser):
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--active-split", type=Path, default=DEFAULT_ACTIVE_SPLIT)
    parser.add_argument(
        "--build-dir", type=Path, default=Path("cmake-build-relwithdebinfo")
    )
    parser.add_argument(
        "--conf", type=Path, default=Path("tools/sbsbench/bench.conf")
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--depth-state-cache-root", type=Path,
        default=DEFAULT_DEPTH_STATE_CACHE_ROOT,
    )
    parser.add_argument(
        "--no-depth-state-cache", action="store_true",
        help="freeze a plan with cross-workspace depth-state reuse disabled",
    )
    parser.add_argument(
        "--scored-result-cache-root", type=Path,
        default=DEFAULT_SCORED_RESULT_CACHE_ROOT,
    )
    parser.add_argument(
        "--no-scored-result-cache", action="store_true",
        help="freeze a plan with compacted scored-result reuse disabled",
    )
    parser.add_argument("--run-prefix", default="ordv2-target-only-v9")
    parser.add_argument("--production", action="append")
    parser.add_argument("--clip-limit", type=int)
    parser.add_argument(
        "--score-workers", type=int, default=ordinal.DEFAULT_SCALE_SCORE_JOBS
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="freeze and print the exact work plan")
    _common_plan_arguments(plan)
    run = commands.add_parser("run", help="run or resume a frozen plan")
    run.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    run.add_argument("--accept-plan-sha256", required=True)
    run.add_argument("--stop-after", choices=ordinal.PHASES, default="catalog")
    run.add_argument("--repair-partials", action="store_true")
    status = commands.add_parser("status", help="show read-only progress")
    status.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    status.add_argument("--verify-completed", action="store_true")
    status.add_argument(
        "--workers", type=int,
        help="bounded verification workers (default: frozen score workers)",
    )
    verify = commands.add_parser("verify", help="revalidate corpus publications")
    verify.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    verify.add_argument("--completed-only", action="store_true")
    verify.add_argument(
        "--workers", type=int,
        help="bounded verification workers (default: frozen score workers)",
    )
    args = parser.parse_args(argv)
    if getattr(args, "score_workers", 1) < 1:
        parser.error("--score-workers must be positive")
    if getattr(args, "clip_limit", None) is not None and args.clip_limit < 1:
        parser.error("--clip-limit must be positive")
    if getattr(args, "workers", None) is not None and args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        result = {
            "plan": command_plan,
            "run": command_run,
            "status": command_status,
            "verify": command_verify,
        }[args.command](args)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ordinal regeneration failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
