import argparse
import os
import itertools as it
import time
import subprocess

parser = argparse.ArgumentParser(description='Submit jobs')
parser.add_argument('-alg_name', action = 'store', dest='algs', type=str, default = 'NSGA2', help='Algorithm to use')
parser.add_argument('-S', action = 'store', dest='Ss', type=str, default = '100,200,300,400,500,600,700,800,900,1000', help='Population size')
parser.add_argument('-dim', action = 'store', dest='dims', type=str, default = '5,15,25,50,75,100,125,150,175,200', help='Number of objectives/variables')
parser.add_argument('-n_gen', action = 'store', dest='n_gens', type=str, default = '100', help='Number of generations')
parser.add_argument('-diagnostic_id', action = 'store', dest='diagnostic_ids', type=str, default = '5', help='Diagnostic problem id')
parser.add_argument('-L', action = 'store', dest='Ls', type=str, default = '10', help='Search space limit')
parser.add_argument('-damp', action = 'store', dest='damps', type=str, default = '1.0', help='Dampening factor')
# parser.add_argument('-epsilon', action = 'store', dest='eps', type=str, default = '0.0', help='Epsilon value')
# parser.add_argument('-epsilon_type', action = 'store', dest='ep_types', type=str, default = '0', help='Epsilon type')
parser.add_argument('-seed', action = 'store', dest='seeds', type=str, default = '14724,24284,31658,6933,1318,16695,27690,8233,24481,6832,13352,4866,12669,12092,15860,19863,6654,10197,29756,14289', help='Random seed')
parser.add_argument('-rdir', type=str, default = '/mnt/scratch/shahban1/MOO/', help='Results directory')
parser.add_argument('-n_trials', action='store', dest='N_TRIALS', default=20, type=int, help='Number of trials to run')
parser.add_argument('-n_jobs', action='store', default=1, type=int, help='Number of parallel jobs')
parser.add_argument('-mem', action='store', dest='mem', default=10000, type=int, help='memory request and limit (MB)')
parser.add_argument('--slurm', action='store_true', default=False, help='Run on an slurm HPC')
parser.add_argument('-time', action='store', dest='time', default='72:00:00', type=str, help='time in HR:MN:SS')
args = parser.parse_args()

n_trials = len(args.seeds)
seeds = args.seeds.split(',')[:n_trials]
algs = args.algs.split(',')
Ss = args.Ss.split(',')
dims = args.dims.split(',')
n_gens = args.n_gens.split(',')
diagnostic_ids = args.diagnostic_ids.split(',')
Ls = args.Ls.split(',')
damps = args.damps.split(',')
# eps = args.eps.split(',')
# ep_types = args.ep_types.split(',')
args.slurm = True

# write run commands
all_commands = []
job_info = []
rdir = '/'.join([args.rdir])
os.makedirs(rdir, exist_ok=True)

for alg,s,dim,damp,seed,epsilon,epsilon_type,n_gen,diagnostic_id,l in it.product(algs,Ss,dims,damps,seeds,n_gens,diagnostic_ids,Ls):

	all_commands.append(
		f'python /mnt/home/shahban1/MultiObjectivePrediction/single_comparison_experiment.py -alg_name {alg} -S {int(s)} -dim {int(dim)} -damp {float(damp)} -seed {int(seed)} -rdir {rdir} -n_gen {int(n_gen)} -diagnostic_id {int(diagnostic_id)} -L {int(l)}'
	)

	job_info.append({
		#'alg':alg,
         'S':s,
         'dim':dim,
         'damp':damp,  
         'epsilon': epsilon, 
		 'epsilon_type': epsilon_type,
		 'seed': seed,
		 'rdir': rdir
        
	})


print(len(job_info), 'total jobs created')

if args.slurm:
	# write a jobarray file to read commans from
	jobarrayfile = 'jobfiles/joblist.txt'
	os.makedirs('jobfiles', exist_ok=True)
	for i, run_cmd in enumerate(all_commands):

		job_name = '_'.join([x + '-' + f'{job_info[i][x]}' for x in
							['alg_name','S','dim', 'diagnostic_id', 'seed']])
		job_file = f'jobfiles/{job_name}.sb'
		out_file = job_info[i]['rdir'] + '/' + job_name + '_%J.out'

		batch_script = (
			f"""#!/usr/bin/bash 
#SBATCH -A ecode
#SBATCH --output={out_file} 
#SBATCH --job-name={job_name} 
#SBATCH --ntasks={1} 
#SBATCH --cpus-per-task={1} 
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}

date
source /mnt/home/shahban1/lexicase-tradeoffs/lex_env/bin/activate

{run_cmd}

date
"""
		)

		with open(job_file, 'w') as f:
			f.write(batch_script)

		print(run_cmd)
		# sbatch_response = subprocess.check_output(
		# 	[f'sbatch {job_file}'], shell=True).decode()     # submit jobs
		# print(sbatch_response)
