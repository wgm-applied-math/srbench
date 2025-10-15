"""
Job Submission
--------------

This script automates the execution of machine learning experiments on multiple
datasets. It can run locally or submit jobs to a SLURM cluster using Singularity
containers.

Features:
- Avoids rerunning completed experiments unless --noskips is specified.
- Supports adding noise, scaling, and symbolic dataset handling.
- Parallel local execution with joblib.

Usage:
    python run_experiments.py <dataset_dir> [options]

Example:
    python run_experiments.py pmlb/datasets -ml sklearn_rf,afp_ehc --slurm -n_jobs 4

"""

import os
import sys
import shlex
import argparse
import subprocess
import pandas as pd
import numpy as np
from glob import glob
from joblib import Parallel, delayed
from yaml import load, Loader
from seeds import SEEDS


def parse_args():
    """Parse command line arguments for experiment configuration."""

    parser = argparse.ArgumentParser(
        description=("Run and manage machine learning experiments locally or on SLURM clusters. "
                     "Use -h/--help to see how to configure this file."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=True
    )

    # Script name
    parser.add_argument("-script", dest="script", type=str, default="optimize_model", help="Name of Python script to execute inside container.")

    # Directories and paths
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory (e.g., pmlb/datasets).")
    parser.add_argument("-pretrained_dir", dest="pretrained_dir", type=str, default="/", help="Folder with pre-trained models or checkpoints.")
    parser.add_argument("-results", dest="results_dir", type=str, default="results", help="Results output directory.")
    parser.add_argument("-images", dest="singularity_dir", type=str, default="../singularity", help="Directory containing Singularity (.sif) images. Required when using SLURM.")

    # Experiment configuration
    parser.add_argument("-ml", dest="learners", type=str, default=None, help="Comma-separated list of ML methods (must match filenames in experiment/methods).")
    parser.add_argument("--sym_data", action="store_true", help="Specify if datasets are symbolic.")
    parser.add_argument("--save_population", action="store_true", help="Save population at the end of the run.")
    parser.add_argument("--skip_tuning", action="store_true", help="Skip hyperparameter tuning step.")
    parser.add_argument("--tuned", action="store_true", help="Run tuned version of estimators.")
    parser.add_argument("--scale_x", action="store_true", help="Apply feature scaling.")
    parser.add_argument("--scale_y", action="store_true", help="Apply target scaling.")
    parser.add_argument('--ecotracker', action='store_true', default=False, help='Enables ecotracker. May conflict with SO.')

    # Noise injection
    parser.add_argument("-target_noise", dest="y_noise", type=float, default=0.0, help="Gaussian noise added to target variable.")
    parser.add_argument("-feature_noise", dest="x_noise", type=float, default=0.0, help="Gaussian noise added to input features.")

    # Execution mode and resources
    parser.add_argument("--no_docker", action="store_true", help="Run jobs locally without using docker images.")
    parser.add_argument("--local", action="store_true", help="Run jobs locally.")
    parser.add_argument("--slurm", action="store_true", help="Submit jobs to SLURM cluster.")

    # Resources
    parser.add_argument("-n_jobs", type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument("-fit_time_limit", dest="fit_time", type=int, default=3600, help="Time limit (in seconds) for model fitting.")
    parser.add_argument("-job_time_limit", dest="job_time", type=str, default="8:00", help="Maximum job walltime (HH:MM).")

    # Reproducibility
    parser.add_argument("-seed", dest="seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("-n_trials", dest="n_trials", type=int, default=30, help="Number of trials per learner.")
    parser.add_argument("-starting_seed", dest="start_seed", type=int, default=0, help="Starting seed index.")

    # SLURM-specific arguments. Ignored when running locally
    parser.add_argument("--noskips", action="store_true", help="Overwrite existing results.")
    parser.add_argument("--test", action="store_true", help="Run in test mode with minimal configuration.")
    parser.add_argument("-job_limit", type=int, default=5000, help="Maximum number of concurrent jobs.")
    parser.add_argument("-max_samples", type=int, default=40000, help="Limit number of training samples.")
    parser.add_argument("-m", type=int, default=10000, help="Memory allocation per job (MB).")
    parser.add_argument("-q", type=str, default='', help="SLURM queue.")

    return parser.parse_args()


def detect_datasets(dataset_dir):
    """Return sorted list of dataset file paths based on dataset size."""

    if dataset_dir.endswith(".tsv.gz"):
        datasets = [dataset_dir]
    else:
        datasets = glob(os.path.join(dataset_dir, "*/*.tsv.gz"))

    dataset_sizes = []
    for dataset in datasets:
        # grab regression datasets
        metadata = load(
            open('/'.join(dataset.split('/')[:-1])+'/metadata.yaml','r'),
                Loader=Loader)
        if metadata['task'] != 'regression':
            print(f"Skipping non-regression dataset: {dataset}")
            continue
        
        dataset_sizes.append(metadata["n_instances"] * metadata["n_features"])

    # Sort datasets by datapoints (so faster jobs get submitted first)
    return [datasets[i] for i in np.argsort(dataset_sizes)]


def detect_learners(args):
    """Determine list of learners to use."""

    if args.learners is None:
        return [ml.split("/")[-2] for ml in glob("experiment/methods/*/regressor.py", recursive=True)]
    
    return [
        ("tuned" if (args.tuned or args.script == "optimize_model") else "") + ml
        for ml in args.learners.split(",")
    ]


def run_no_docker(commands, n_jobs):
    """Run all commands locally in parallel. Several processess will be spawned!"""

    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)


def run_local(commands, job_info, args):
    """Run experiment commands locally using Docker containers in parallel."""

    print("Running locally with Docker...")

    def _run(cmd, ml):
        docker_cmd = [
            "docker", "compose", "run", "--rm", 
            "-v", f"{os.getcwd()}/experiment:/srbench",
            "-v", f"{os.getcwd()}/{args.dataset_dir}:/{args.dataset_dir}",
            "-v", f"{os.getcwd()}/{args.results_dir}:/{args.results_dir}",
            "-v", f"{args.pretrained_dir}:/srbench_pretrained",
            f"{ml.replace('tuned', '').lower()}",
            "python", "-u"
        ] + shlex.split(f"/srbench/{cmd}")

        print(" ".join(docker_cmd))
        subprocess.run(docker_cmd)

    Parallel(n_jobs=args.n_jobs)(delayed(_run)(cmd, info['ml'])
                                 for cmd, info in zip(commands, job_info))


def run_slurm(commands, job_info, args):
    """
    Submit jobs to a SLURM cluster.

    This function first queries the active SLURM job queue using `squeue`
    to identify already running or pending jobs (to avoid resubmission).
    Then, it generates and submits batch scripts for remaining jobs.
    """

    # ----------------------------------------------------------
    # Query existing SLURM jobs
    # ----------------------------------------------------------
    try:
        print("Checking for existing SLURM jobs...")
        res = subprocess.check_output(['squeue -o %j'], shell=True)
        current_jobs = res.decode().split('\n')
        queued_jobs  = []
        for i, cmd in enumerate(commands):
            job_name = f"{job_info[i]['dataset']}_{job_info[i]['ml']}_{job_info[i]['seed']}_{args.script}"

            if args.y_noise>0: job_name += '_target-noise'+str(args.y_noise)
            if args.x_noise>0: job_name += '_feature-noise'+str(args.x_noise)
        
            if job_name in current_jobs:
                print(f"Skipping already queued job: {job_name}")
                queued_jobs.append(i)
                continue
            
        commands = [cmd for (i, cmd) in enumerate(commands) if not i in queued_jobs ]
        job_info = [job for (i, job) in enumerate(job_info) if not i in queued_jobs ]
    except subprocess.CalledProcessError:
        print("Warning: could not retrieve job list from SLURM.")
        queued_jobs = []

    print('skipped', len(queued_jobs),'queued jobs. Override with --noskips.')

    max_jobs = args.job_limit - len(queued_jobs)
    if len(commands) > max_jobs:
        print(f'Shaving jobs down to job limit ({args.job_limit}) minus queued jobs.')
        commands = commands[:max_jobs]

    input()
    # ----------------------------------------------------------
    # Submit new jobs
    # ----------------------------------------------------------
    gpu_setting = "#SBATCH --gres=gpu:1" if 'gpu' in args.q else ""
    for i, cmd in enumerate(commands):
        job_name = f"{job_info[i]['dataset']}_{job_info[i]['ml']}_{job_info[i]['seed']}_{args.script}"

        if args.y_noise>0: job_name += '_target-noise'+str(args.y_noise)
        if args.x_noise>0: job_name += '_feature-noise'+str(args.x_noise)
    
        out_file = f"{job_info[i]['results_path']}/{job_info[i]['dataset']}/{job_name}.%J.out"
        err_file = out_file.replace('.out', '.err')

        singularity_cmd = (
            "singularity run --no-home --contain "
            f"--bind $(pwd)/experiment:/srbench,$(pwd)/{args.dataset_dir}:/{args.dataset_dir} "
            f"--bind {args.pretrained_dir}:/srbench_pretrained/ "
            f"--bind $(pwd)/{args.results_dir}:/{args.results_dir} --fakeroot --writable-tmpfs "
            f"{args.singularity_dir}/{job_info[i]['ml'].replace('tuned', '')}.sif "
            f"python /srbench/{cmd}"
        )

        batch_script_lines = [
            "#!/usr/bin/bash",
            f"#SBATCH -o {out_file}",
            f"#SBATCH --error={err_file}",
            "#SBATCH -N 1",
            f"#SBATCH -n {args.n_jobs}",
            f"#SBATCH -J {job_name}",
            f"#SBATCH -p {args.q}",
            f"#SBATCH --ntasks-per-node=1 --time={args.job_time}:00",
            f"#SBATCH --mem-per-cpu={args.m}",
            gpu_setting,
            "hostname",
            "",
            "TZ='America/New_York'",
            "export TZ",
            "",
            "timedatectl",
            "",
            singularity_cmd,
        ]
        batch_script = "\n".join(batch_script_lines)
        with open("tmp_slurm_script", "w") as f:
            f.write(batch_script)

        print(f"Submitting job: {job_name}")
        try:
            sbatch_response = subprocess.check_output(["sbatch tmp_slurm_script"], shell=True).decode()
            print(f"Submitted: {sbatch_response.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job {job_name}: {e}")


def main():
    args = parse_args()
     
    print('script:', args.script)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Results directory: {args.results_dir}")

    learners = detect_learners(args)
    datasets = detect_datasets(args.dataset_dir)

    print(f"Found {len(datasets)} datasets and {len(learners)} learners.")

    if args.y_noise > 0: print('using target-noise', str(args.y_noise))
    if args.x_noise > 0: print('using feature-noise', str(args.x_noise))

    all_commands, job_info, jobs_w_results = [], [], []

    for t in range(args.start_seed, args.start_seed + args.n_trials):
        random_state = args.seed if args.seed and args.n_trials == 1 else SEEDS[t]

        for dataset in datasets:
            dataname = os.path.basename(dataset).replace(".tsv.gz", "")
            results_path = os.path.join(args.results_dir, dataname)
            os.makedirs(results_path, exist_ok=True)

            if (not args.sym_data and 'rethinking_feynman' in dataname):
                continue

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            for ml in learners:
                save_file = f"{results_path}/{dataname}_{ml}_{random_state}"

                if not args.noskips:
                    suffix = ''
                    if args.y_noise>0:
                        suffix += '_target-noise'+str(args.y_noise)
                    if args.x_noise>0:
                        suffix += '_feature-noise'+str(args.x_noise)
                    if 'optimize' in args.script:
                        suffix += '_cv_results'
                    suffix += '.json'
                        
                    if os.path.exists(save_file+suffix):
                        jobs_w_results.append([save_file, 'exists'])
                        continue

                # cant have 'tuned' prefix here because it will import the exact name
                cmd = (
                    f"{args.script}.py /{dataset} -ml {ml.replace('tuned', '')} "
                    f"-results_path /{results_path} -seed {random_state} "
                    f"-target_noise {args.y_noise} -feature_noise {args.x_noise} "
                    f"-fit_time_limit {args.fit_time} -max_samples {args.max_samples} "
                    f"{'--scale_x' if args.scale_x else ''} "
                    f"{'--scale_y' if args.scale_y else ''} "
                    f"{'--sym_data' if args.sym_data else ''} "
                    f"{'--skip_tuning' if args.skip_tuning else ''} "
                    f"{'--tuned' if args.tuned else ''}"
                    f"{'--ecotracker' if args.ecotracker else ''} "
                    f"{'--save_population' if args.save_population else ''} "
                    f"{'--test' if args.test else ''} "
                )
                all_commands.append(cmd)
                job_info.append({"ml": ml, "dataset": dataname, "seed": random_state, "results_path": results_path})

    print(f"Prepared {len(all_commands)} jobs...")

    if not args.noskips:
        print('skipped', len(jobs_w_results),'jobs with results. Override with --noskips.')
    
    input("Press Enter to continue")

    if args.no_docker:
        run_no_docker(all_commands, args.n_jobs)
    if args.local:
        run_local(all_commands, job_info, args)
    elif args.slurm:
        run_slurm(all_commands, job_info, args)
    else:
        print("Specify either --local or --slurm to execute jobs.")

    print("Finished submitting jobs.")


if __name__ == '__main__':
    main()
