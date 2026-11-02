'''
Repeat every env many times, one run at a time, on an otherwise idle GPU, for runtime and quality statistics.

Rounds of --seeds-per-round seeds (round r = seeds 10r .. 10r+9), interleaved over the envs, until --until
(local time; runs still going then are killed and not counted) or --max-runs per job. A watcher polls nvidia-smi: a run only starts
once no other process is on the GPU, and if one shows up during a run, the run is killed and queued again.
Resumable: (job, seed) pairs already in runs.jsonl are skipped.

    python xnew_rrt/bench_envs.py --until 09:00
    -> <out>/runs.jsonl (one row per finished run), <out>/summary.md / summary.json, <out>/bench.log

Block puzzles get --seed too: their high level is deterministic, the low level's headings are random.
'''

import argparse, datetime, json, os, signal, statistics, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = "checkpoints/spin8_manual.json"
RRT_ENVS = ["1a", "1b", "1c", "1c_c100", "2a", "2b", "3a", "3b", "4a", "4b", "4b_r50", "4c", "4d"]
PUZZLES = {"5a": [], "5b": [], "5c": [], "5d": [], "5c_a15": ["--max-final-angle", "15"]}


def gpu_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return {int(p) for p in out.split() if p.strip().isdigit()}


def descendants(pid):
    pids, todo = {pid}, [pid]
    while todo:
        p = todo.pop()
        try:
            with open(f"/proc/{p}/task/{p}/children") as f:
                kids = [int(k) for k in f.read().split()]
        except OSError:
            kids = []
        pids.update(kids)
        todo += kids
    return pids


def command(job, seed, out, args):
    py = [os.path.join(REPO, "xnew_rrt", "py.sh")]
    if job in PUZZLES:
        env = job.split("_")[0]
        return py + ["xnew_rrt/block_puzzle.py", f"envs/{env}.json", "--checkpoint", CK, "--no-render",
                     "--max-attempts", str(args.max_attempts), "--seed", str(seed), "--out", out] + PUZZLES[job]
    return py + ["xnew_rrt/run_envs.py", f"envs/{job}.json", "--checkpoint", CK, "--no-render", "--seed", str(seed),
                 "--max-edges", str(args.max_edges), "--time-limit", str(args.time_limit), "--out", out]


def metrics(job, out):
    env = job.split("_")[0] if job in PUZZLES else job
    try:
        with open(os.path.join(out, "results", f"{env}.result.json")) as f:
            r = json.load(f)
    except OSError:
        return None
    if job in PUZZLES:
        pushes, attempts = r["pushes"], r["attempts"]
        return dict(success=r["solved"], seconds=r["seconds"], pushes=len(pushes), push_searches=len(attempts),
                    candidates=sum(a.get("candidates", 0) for a in attempts),
                    trials=sum(a.get("trials", 0) for a in attempts),
                    final_err_mm=pushes[-1]["err_mm"] if pushes else None,
                    max_angle_deg=max((p["max_angle_deg"] for p in pushes), default=None),
                    flagged=sum(bool(p["flagged"]) for p in pushes))
    return dict(success=r["reached"], seconds=r["seconds"], goal_distance_mm=r["goal_distance_mm"], edges=r["edges"],
                edges_to_goal=r["edges_to_goal"], seconds_to_goal=r["seconds_to_goal"], nodes=r["nodes"],
                iterations=r["iterations"], invalid=r["invalid"], pruned=r["pruned"], resets=r["resets"],
                pushes=r["pushes"], controls=len(r["controls"]),
                solution_steps=sum(int(c[1]) for c in r["controls"]) if r["controls"] and len(r["controls"][0]) > 1 else None,
                stopped=r["stopped"])


def summarize(rows, out_dir, jobs):
    by_job = {}
    for row in rows:
        by_job.setdefault(row["job"], []).append(row)
    stats = {}
    keys = ["wall_s", "seconds", "seconds_to_goal", "edges", "edges_to_goal", "nodes", "goal_distance_mm", "controls",
            "solution_steps", "resets", "pushes", "push_searches", "candidates", "trials", "final_err_mm", "max_angle_deg"]
    for job in jobs:
        rs = by_job.get(job, [])
        s = dict(runs=len(rs), success=sum(bool(r.get("success")) for r in rs))
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None and (k not in ("seconds_to_goal", "edges_to_goal") or r.get("success"))]
            if vals:
                s[k] = dict(mean=statistics.fmean(vals), median=statistics.median(vals),
                            std=statistics.stdev(vals) if len(vals) > 1 else 0.0, min=min(vals), max=max(vals))
        stats[job] = s
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(stats, f, indent=1)

    def ms(s, k, fmt="{:.0f}"):
        return f"{fmt.format(s[k]['mean'])} ± {fmt.format(s[k]['std'])}" if k in s else "-"
    lines = [f"# Benchmark ({CK}, one run at a time, GPU otherwise idle)", "",
             f"updated {datetime.datetime.now():%Y-%m-%d %H:%M}; mean ± std over runs; to-goal columns over successful runs", "",
             "| env | success | wall [s] | planner [s] | to goal [s] | edges | edges to goal | nodes | goal dist [mm] | controls | resets |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for job in jobs:
        if job in PUZZLES:
            continue
        s = stats[job]
        lines.append(f"| {job} | {s['success']}/{s['runs']} | {ms(s, 'wall_s')} | {ms(s, 'seconds')} | {ms(s, 'seconds_to_goal')} | "
                     f"{ms(s, 'edges')} | {ms(s, 'edges_to_goal')} | {ms(s, 'nodes')} | {ms(s, 'goal_distance_mm', '{:.1f}')} | "
                     f"{ms(s, 'controls', '{:.1f}')} | {ms(s, 'resets', '{:.1f}')} |")
    lines += ["", "| puzzle | solved | wall [s] | planner [s] | pushes | push searches | candidate vines | final err [mm] | max angle [deg] |",
              "|---|---|---|---|---|---|---|---|---|"]
    for job in jobs:
        if job not in PUZZLES:
            continue
        s = stats[job]
        lines.append(f"| {job} | {s['success']}/{s['runs']} | {ms(s, 'wall_s')} | {ms(s, 'seconds')} | {ms(s, 'pushes', '{:.1f}')} | "
                     f"{ms(s, 'push_searches', '{:.1f}')} | {ms(s, 'candidates')} | {ms(s, 'final_err_mm', '{:.1f}')} | "
                     f"{ms(s, 'max_angle_deg', '{:.1f}')} |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    global CK
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", nargs="*", default=RRT_ENVS + list(PUZZLES))
    parser.add_argument("--until", default="09:00", help="local HH:MM; the next one from now")
    parser.add_argument("--seeds-per-round", type=int, default=10)
    parser.add_argument("--max-runs", type=int, default=None, help="stop once every job has this many runs")
    parser.add_argument("--job-max-runs", nargs="*", default=[], help="per-job caps, e.g. 3b=5")
    parser.add_argument("--max-edges", type=int, default=300000)
    parser.add_argument("--time-limit", type=float, default=20.0, help="[min] per RRT run")
    parser.add_argument("--max-attempts", type=int, default=400, help="push searches per puzzle run")
    parser.add_argument("--idle-wait", type=float, default=60.0, help="[s] after another process was seen, the GPU must be free this long before a run")
    parser.add_argument("--poll", type=float, default=2.0, help="[s] watcher period")
    parser.add_argument("--out", default="out/bench_overnight")
    parser.add_argument("--checkpoint", default=None, help=f"default {CK}")
    args = parser.parse_args()
    CK = args.checkpoint or CK
    os.chdir(REPO)

    now = datetime.datetime.now()
    hh, mm = map(int, args.until.split(":"))
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline <= now:
        deadline += datetime.timedelta(days=1)
    deadline = deadline.timestamp()

    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "runs.jsonl")
    logf = open(os.path.join(args.out, "bench.log"), "a")

    def log(msg):
        line = f"{datetime.datetime.now():%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    rows = []
    if os.path.exists(rows_path):
        with open(rows_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    done = {(r["job"], r["seed"]) for r in rows}
    job_max = {k: int(v) for k, v in (kv.split("=") for kv in args.job_max_runs)}
    log(f"start: {len(rows)} runs already done, deadline {datetime.datetime.fromtimestamp(deadline):%m-%d %H:%M}")

    def wait_idle():
        # the idle wait only applies once another process has been seen; a free GPU starts the next run at once
        free_since, seen = None, False
        while time.time() < deadline:
            others = gpu_pids()
            if not others:
                free_since = free_since or time.time()
                if not seen or time.time() - free_since >= args.idle_wait:
                    return True
            else:
                seen = True
                if free_since is not None or not hasattr(wait_idle, "said"):
                    log(f"waiting: GPU in use by {sorted(others)}")
                    wait_idle.said = True
                free_since = None
            time.sleep(args.poll)
        return False

    r = 0
    while time.time() < deadline and (args.max_runs is None or r * args.seeds_per_round < args.max_runs):
        for seed in range(r * args.seeds_per_round, (r + 1) * args.seeds_per_round):
            for job in args.jobs:
                cap = job_max.get(job, args.max_runs)
                if cap is not None and sum(j == job for j, _ in done) >= cap:
                    continue
                while (job, seed) not in done:
                    if not wait_idle():
                        log("deadline reached while waiting")
                        summarize(rows, args.out, args.jobs)
                        return
                    out = os.path.join(args.out, "runs", job, f"s{seed:03d}")
                    os.makedirs(out, exist_ok=True)
                    t0 = time.time()
                    with open(os.path.join(out, "run.log"), "w") as f:
                        proc = subprocess.Popen(command(job, seed, out, args), stdout=f, stderr=subprocess.STDOUT,
                                                start_new_session=True)
                    intruders, killed = set(), None
                    load0 = os.getloadavg()[0]
                    while proc.poll() is None:
                        ours = descendants(proc.pid)
                        new = gpu_pids() - ours
                        if new:
                            intruders, killed = new, "contended"
                        elif time.time() > deadline:
                            killed = "deadline"
                        if killed:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                            break
                        time.sleep(args.poll)
                    wall = time.time() - t0
                    if killed == "deadline":
                        log(f"{job} s{seed}: killed at the deadline after {wall:.0f} s (not counted)")
                        summarize(rows, args.out, args.jobs)
                        return
                    if killed == "contended":
                        log(f"{job} s{seed}: killed after {wall:.0f} s, other GPU processes {sorted(intruders)}; requeued")
                        continue
                    m = metrics(job, out)
                    row = dict(job=job, seed=seed, round=r, start=datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
                               wall_s=wall, exit=proc.returncode, loadavg_start=load0, loadavg_end=os.getloadavg()[0],
                               **(m or dict(success=False, error="no result.json")))
                    rows.append(row)
                    done.add((job, seed))
                    with open(rows_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    summarize(rows, args.out, args.jobs)
                    log(f"{job} s{seed}: exit {proc.returncode}, {wall:.0f} s, success {row.get('success')}"
                        + (f", {row['edges']} edges" if "edges" in row else "")
                        + (f", {row['candidates']} candidates" if "candidates" in row else ""))
        log(f"round {r} done")
        r += 1
    summarize(rows, args.out, args.jobs)


if __name__ == "__main__":
    main()
