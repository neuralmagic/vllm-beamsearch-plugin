"""Stress a running beam-search plugin server and sample memory drift."""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path
import random
import subprocess
import time
from collections import Counter, defaultdict
from typing import Any

import httpx

PROMPTS = [
    "I don't think so. Can I get you my show show and you can check and see?",
    "Yeah, um, fire 17. $62 and was took it taken on my account.",
    "Okay. I'm going to. Thank you. Okay. Thank you. 4906 Chicago.",
    "Hello? No. Hello. Yes, it's a more time because I'm in.",
    "You can see all my history the balance and when I added up.",
    "I keep I don't understand this. Is it a waltzing?",
    "Oh, and before you, before we live before. I forget.",
    "Okay, so that's what. So go ahead, all right.",
]


def _request_body(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    beam_width: int,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "add_special_tokens": False,
        "vllm_xargs": {
            "beam_width": beam_width,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        },
    }


def _proc_cmdline(pid: int) -> list[str]:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in data.split(b"\0") if part]


def _proc_ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _all_pids() -> list[int]:
    return [
        int(path.name)
        for path in Path("/proc").iterdir()
        if path.name.isdigit()
    ]


def _server_roots(port: int) -> list[int]:
    roots = []
    for pid in _all_pids():
        cmd = _proc_cmdline(pid)
        if not cmd:
            continue
        if "vllm.entrypoints.openai.api_server" not in " ".join(cmd):
            continue
        if "--port" in cmd:
            idx = cmd.index("--port")
            if idx + 1 < len(cmd) and cmd[idx + 1] == str(port):
                roots.append(pid)
                continue
        if f"--port={port}" in cmd:
            roots.append(pid)
    return roots


def _descendants(roots: list[int]) -> set[int]:
    children: dict[int, list[int]] = defaultdict(list)
    for pid in _all_pids():
        ppid = _proc_ppid(pid)
        if ppid is not None:
            children[ppid].append(pid)

    seen = set(roots)
    stack = list(roots)
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def _rss_mb(pids: set[int]) -> float:
    total_kb = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kb += int(line.split()[1])
                    break
        except OSError:
            pass
    return total_kb / 1024.0


def _gpu_mb(pids: set[int]) -> float:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return -1.0

    total = 0.0
    for line in out.splitlines():
        if not line.strip():
            continue
        pid_s, mem_s = [part.strip() for part in line.split(",", 1)]
        try:
            if int(pid_s) in pids:
                total += float(mem_s)
        except ValueError:
            continue
    return total


def memory_sample(port: int) -> dict[str, Any]:
    roots = _server_roots(port)
    pids = _descendants(roots)
    return {
        "time_s": time.time(),
        "roots": len(roots),
        "pids": len(pids),
        "rss_mb": _rss_mb(pids),
        "gpu_mb": _gpu_mb(pids),
    }


async def health(client: httpx.AsyncClient, base_url: str) -> bool:
    try:
        resp = await client.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


async def send_one(
    client: httpx.AsyncClient,
    endpoint: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[str, float]:
    start = time.perf_counter()
    try:
        resp = await client.post(endpoint, json=body, timeout=timeout)
        elapsed = time.perf_counter() - start
        if resp.status_code != 200:
            return f"http_{resp.status_code}", elapsed
        payload = resp.json()
        if payload.get("choices"):
            return "ok", elapsed
        return "bad_payload", elapsed
    except httpx.ReadTimeout:
        return "timeout", time.perf_counter() - start
    except Exception as exc:
        return type(exc).__name__, time.perf_counter() - start


async def run_batch(
    client: httpx.AsyncClient,
    endpoint: str,
    args: argparse.Namespace,
    *,
    requests: int,
    concurrency: int,
    timeout: float,
) -> Counter[str]:
    sem = asyncio.Semaphore(concurrency)

    async def task() -> str:
        prompt = random.choice(PROMPTS)
        body = _request_body(
            prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            beam_width=args.beam_width,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        async with sem:
            status, _elapsed = await send_one(client, endpoint, body, timeout)
            return status

    return Counter(await asyncio.gather(*(task() for _ in range(requests))))


async def abort_storm(
    client: httpx.AsyncClient,
    endpoint: str,
    args: argparse.Namespace,
    *,
    requests: int,
) -> Counter[str]:
    async def task() -> str:
        body = _request_body(
            random.choice(PROMPTS),
            model=args.model,
            max_tokens=args.max_tokens,
            beam_width=args.beam_width,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        timeout_ms = random.randint(args.abort_min_ms, args.abort_max_ms)
        status, _elapsed = await send_one(
            client, endpoint, body, timeout_ms / 1000.0
        )
        return "aborted" if status == "timeout" else status

    return Counter(await asyncio.gather(*(task() for _ in range(requests))))


def _drift(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    width = max(1, len(values) // 5)
    first = sum(values[:width]) / width
    last = sum(values[-width:]) / width
    return last - first


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8005")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--model", default="llama3-8b")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--requests-per-round", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--abort-rounds", type=int, default=0)
    parser.add_argument("--abort-requests", type=int, default=64)
    parser.add_argument("--abort-min-ms", type=int, default=30)
    parser.add_argument("--abort-max-ms", type=int, default=150)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--csv", default="/tmp/vllm_beam_stress_memory.csv")
    args = parser.parse_args()

    endpoint = f"{args.base_url}/v1/completions"
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=180.0) as client:
        if not await health(client, args.base_url):
            raise SystemExit(f"server is not healthy at {args.base_url}")

        sample = memory_sample(args.port)
        sample.update({"round": 0, "requests": 0, "phase": "start"})
        rows.append(sample)
        print(
            "start: "
            f"rss={sample['rss_mb']:.1f}MB gpu={sample['gpu_mb']:.1f}MB "
            f"pids={sample['pids']}"
        )

        completed = 0
        for round_idx in range(1, args.rounds + 1):
            counts = await run_batch(
                client,
                endpoint,
                args,
                requests=args.requests_per_round,
                concurrency=args.concurrency,
                timeout=180.0,
            )
            totals.update(counts)
            completed += args.requests_per_round

            if round_idx % args.sample_every == 0 or round_idx == args.rounds:
                sample = memory_sample(args.port)
                sample.update({
                    "round": round_idx,
                    "requests": completed,
                    "phase": "load",
                })
                rows.append(sample)
                print(
                    f"round={round_idx:4d} reqs={completed:6d} "
                    f"ok={totals['ok']:6d} errors={sum(totals.values()) - totals['ok']:4d} "
                    f"rss={sample['rss_mb']:8.1f}MB "
                    f"gpu={sample['gpu_mb']:8.1f}MB"
                )

        for round_idx in range(1, args.abort_rounds + 1):
            counts = await abort_storm(
                client,
                endpoint,
                args,
                requests=args.abort_requests,
            )
            totals.update(counts)
            sample = memory_sample(args.port)
            sample.update({
                "round": args.rounds + round_idx,
                "requests": completed,
                "phase": "abort",
            })
            rows.append(sample)
            print(
                f"abort_round={round_idx:3d} "
                f"aborted={counts['aborted']:4d} ok={counts['ok']:4d} "
                f"rss={sample['rss_mb']:8.1f}MB "
                f"gpu={sample['gpu_mb']:8.1f}MB"
            )

        healthy = await health(client, args.base_url)

    out_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "round",
                "requests",
                "time_s",
                "roots",
                "pids",
                "rss_mb",
                "gpu_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    load_rows = [row for row in rows if row["phase"] in {"start", "load"}]
    rss_drift = _drift([float(row["rss_mb"]) for row in load_rows])
    gpu_drift = _drift([float(row["gpu_mb"]) for row in load_rows])
    total_errors = sum(totals.values()) - totals["ok"] - totals["aborted"]
    elapsed = time.perf_counter() - start
    print("")
    print("summary:")
    print(f"  healthy: {healthy}")
    print(f"  totals: {dict(totals)}")
    print(f"  non-abort errors: {total_errors}")
    print(f"  elapsed_s: {elapsed:.1f}")
    print(f"  rss_drift_mb: {rss_drift:+.1f}")
    print(f"  gpu_drift_mb: {gpu_drift:+.1f}")
    print(f"  csv: {out_path}")

    if not healthy or total_errors:
        raise SystemExit(1)


def main_sync() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
