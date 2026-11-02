#!/bin/bash
# The actvine Python, capped at 70% of the machine's RAM: it runs in its own cgroup, so going over
# the cap kills it (exit 137) instead of swapping the whole machine to a halt. No swap either.
# VRAM is capped at 70% inside the scripts (torch.cuda.set_per_process_memory_fraction).
#   xnew_rrt/py.sh xnew_rrt/run_envs.py envs/1b.json ...
exec systemd-run --user --scope --quiet -p MemoryMax=70% -p MemorySwapMax=0 \
    /home/yitian/miniconda3/envs/actvine/bin/python -B "$@"
