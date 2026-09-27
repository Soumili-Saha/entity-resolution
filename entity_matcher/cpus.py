"""Number of CPU cores this process may use: the scheduler's reservation when there is one.

On shared clusters os.cpu_count() reports every core of the node, not the job's allocation, so
using it oversubscribes the node. Order: N_CPUS override, PBS NCPUS, SLURM_CPUS_PER_TASK,
OMP_NUM_THREADS, the process CPU affinity (Linux), then os.cpu_count() - 2 (laptop).
"""
import os


def n_cpus():
    """Cores available to this job."""
    for var in ("N_CPUS", "NCPUS", "PBS_NCPUS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        v = os.environ.get(var, "")
        if v.isdigit() and int(v) > 0:
            return int(v)
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
        if n < (os.cpu_count() or n):
            return n
    return max(1, (os.cpu_count() or 2) - 2)
