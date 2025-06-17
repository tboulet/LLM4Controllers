

import argparse
import datetime
import os
import subprocess
# arg train
parser = argparse.ArgumentParser(description="Argument parsing for experiment")
parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, help="Use CPU partition")  
parser.add_argument("--v100", action=argparse.BooleanOptionalAction, help="Use V100 GPUs")     # 32G or other
parser.add_argument("--h100", action=argparse.BooleanOptionalAction, help="Use H100 GPUs")     # 80G
parser.add_argument("--a100", action=argparse.BooleanOptionalAction, help="Use A100 GPUs")     # 80G (not available)
parser.add_argument("--cpus-per-task", type=int, default=1, help="Number of CPUs per task")      # 
parser.add_argument("--dev", action=argparse.BooleanOptionalAction, help="Development mode")
parser.add_argument("--n_gpu", type=int, help="Number of GPUs to use", default=1)
parser.add_argument("--hour",  type=int, default=20)

args = parser.parse_args()


current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_name = f"tboulet_{current_datetime}"
def generate_slurm_script(args,job_name):

    list_lines_script = []

    # Set the time limit based on the mode    
    if args.dev:
        hour = 2
    else:
        hour = args.hour
    
    # For each partition, specify account and constrains (if any), qos, and number of CPUs (if applicable)   
    module_load = ""
    if args.cpu:
        list_lines_script.append("#SBATCH --account=imi@cpu")
        if args.dev:
            list_lines_script.append("#SBATCH --qos=qos_cpu-dev")
        else:
            list_lines_script.append("#SBATCH --qos=qos_cpu-t3")
        args.n_gpu = 0
    elif args.v100:
        list_lines_script.append("#SBATCH --account=imi@v100")
        list_lines_script.append("#SBATCH -C v100-32g")
        if args.dev:
            list_lines_script.append("#SBATCH --qos=qos_gpu-dev")
        elif hour <= 20 :
            list_lines_script.append("#SBATCH --qos=qos_gpu-t3")
        else:
            list_lines_script.append("#SBATCH --qos=qos_gpu-t4")
        n_cpu = min(args.n_gpu * 10,40)
    elif args.h100:
        list_lines_script.append("#SBATCH --account=imi@h100")
        list_lines_script.append("#SBATCH -C h100")
        n_cpu = min(int(args.n_gpu * 24),96)
        module_load = "module load arch/h100"
        if args.dev:
            list_lines_script.append("#SBATCH --qos=qos_gpu_h100-dev")
        elif hour <= 20 :
            list_lines_script.append("#SBATCH --qos=qos_gpu_h100-t3")
        else:
            list_lines_script.append("#SBATCH --qos=qos_gpu_h100-t4")

    # elif args.a100:
    #     account = "imi@a100"
    #     c="a100"
    #     n_cpu = min(int(args.n_gpu * 8),64)
    #     module_load = "module load arch/a100"
    else:
        raise ValueError("Please specify a GPU type: --cpu, --v100, --h100, or --a100")
    
    if args.cpus_per_task:
        n_cpu = args.cpus_per_task
    list_lines_script.append(f"#SBATCH --cpus-per-task={n_cpu}")
    script_begin = "\n".join(list_lines_script)
    script = f"""#!/bin/bash
{script_begin}
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{args.n_gpu}
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --hint=nomultithread
#SBATCH --time={hour}:00:00
#SBATCH --output=./out/{job_name}-%A/.out
#SBATCH --error=./out/{job_name}-%A/.err
# set -x

hostname
export TMPDIR=$JOBSCRATCH
module purge
{module_load}
ulimit -c 0
export CORE_PATTERN=/dev/null



source $SCRATCH/venv/bin/activate
cd $WORK/LLM4Controllers

# python run2.py \
#   agent=cg \
#   llm=vllm \
#   > logs/{job_name}.log 2>&1

uvicorn tbllm.run:app --host 0.0.0.0 --port 8000

"""

    return script    


slurmfile_path = f'run_{job_name}.slurm'
full_script = generate_slurm_script(args,job_name)

with open(slurmfile_path, 'w') as f:
    f.write(full_script)

subprocess.call(f'sbatch {slurmfile_path}', shell=True)
# dell 
os.remove(slurmfile_path)