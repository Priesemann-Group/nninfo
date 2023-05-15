from os import cpu_count
import os

# set the maximum number of workers, typically to the number of cpus minus 1
cpus = cpu_count()
if cpus is None:
    raise OSError
elif cpus <= 2:
    N_WORKERS = cpus
else:
    N_WORKERS = cpu_count()
    
# for cluster safe usage uncomment:
CLUSTER_MODE = int(os.environ.get("CLUSTER_MODE", 0)) == 1

if CLUSTER_MODE:
    N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

print(f'{N_WORKERS=}')