

import argparse
import os
import subprocess
# arg train
parser = argparse.ArgumentParser(description="Argument parsing for experiment")
parser.add_argument("--v100", default=True, action=argparse.BooleanOptionalAction, help="Use V100 GPUs")
parser.add_argument("--h100", action=argparse.BooleanOptionalAction, help="Use H100 GPUs")
parser.add_argument("--dev", action=argparse.BooleanOptionalAction, help="Development mode")
parser.add_argument("--long", action=argparse.BooleanOptionalAction, help="long mode 100h instead of 20h")
parser.add_argument("--medium", action=argparse.BooleanOptionalAction, help="medium mode 40h instead of 20h")
parser.add_argument("--n_gpu", type=int, help="Number of GPUs to use", default=1)
parser.add_argument("--hour",  type=int, default=20)

args = parser.parse_args()



job_name ="tboulet"
def generate_slurm_script(args,job_name):

    if args.dev:
        if args.v100:
            dev_script = "#SBATCH --qos=qos_gpu-dev"
        elif args.h100:
            dev_script = "#SBATCH --qos=qos_gpu_h100-dev"
        
        else:
            dev_script = "#SBATCH --qos=qos_gpu_a100-dev"
    else:        
        dev_script = ""

    h = '2' if args.dev else str(args.hour)
    
    if args.long:
        h = '99'
        if args.hour !=20:
            h = str(args.hour)
            
        if args.v100:
            dev_script= "#SBATCH --qos=qos_gpu-t4"
        elif args.h100:
            dev_script = "#SBATCH --qos=qos_gpu_h100-t4"

    if args.medium:
        h = '40'
        if args.v100:
            dev_script= "#SBATCH --qos=qos_gpu-t4"
        elif args.h100:
            dev_script = "#SBATCH --qos=qos_gpu_h100-t4"
    if args.v100:
        account = "imi@v100"
        c = "v100-32g"
        n_cpu = min(args.n_gpu * 10,40)
        module_load = ""
    elif args.h100:
        account = "imi@h100"
        c="h100"
        n_cpu = min(int(args.n_gpu * 24),96)
        module_load = "module load arch/h100"
    else:
        account = "imi@a100"
        c="a100"
        n_cpu = min(int(args.n_gpu * 8),64)
        module_load = "module load arch/a100"
    script = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH -C {c}
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{args.n_gpu}
#SBATCH --cpus-per-task={n_cpu}
{dev_script}
#SBATCH --hint=nomultithread
#SBATCH --time={h}:00:00
#SBATCH --output=./out/{job_name}-%A.out
#SBATCH --error=./out/{job_name}-%A.out
# set -x

export TMPDIR=$JOBSCRATCH
module purge
{module_load}
ulimit -c 0
limit coredumpsize 0
export CORE_PATTERN=/dev/null



source $SCRATCH/venv/bin/activate
cd $WORK/LLM4Controllers
python run.py --config-name=jz agent=random
"""
    return script    


slurmfile_path = f'run_{job_name}.slurm'
full_script = generate_slurm_script(args,job_name)

with open(slurmfile_path, 'w') as f:
    f.write(full_script)

subprocess.call(f'sbatch {slurmfile_path}', shell=True)
# dell 
os.remove(slurmfile_path)