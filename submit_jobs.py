import argparse
import os
import itertools as it


parser = argparse.ArgumentParser(description='Submit jobs')
parser.add_argument('-alg', action = 'store', dest='algs', type=str, default = 'Lexicase', help='Algorithm to use')
parser.add_argument('-S', action = 'store', dest='Ss', type=int, default = 100, help='Population size')
parser.add_argument('-dim', action = 'store', dest='Dims', type=int, default = 5, help='Number of objectives/variables')
parser.add_argument('-n_gen', action = 'store', dest='n_gens', type=int, default = 50000, help='Number of generations')
parser.add_argument('-diagnostic_id', action = 'store', dest='diagnostic_ids', type=int, default = 5, help='Diagnostic problem id')
parser.add_argument('-L', action = 'store', dest='Ls', type=int, default = 100, help='Search space limit')
parser.add_argument('-damp', action = 'store', dest='damps', type=float, default = 1.0, help='Dampening factor')
parser.add_argument('-epsilon', action = 'store', dest='eps', type=float, default = 0.0, help='Epsilon value')
parser.add_argument('-epsilon_type', action = 'store', dest='ep_types', type=int, default = 0, help='Epsilon type')
parser.add_argument('-seed', action = 'store', dest='seeds', type=int, default = 0, help='Random seed')
parser.add_argument('-rdir', type=str, default = 'results', help='Results directory')
parser.add_argument('-n_trials', action='store', dest='N_TRIALS', default=20, type=int, help='Number of trials to run')
parser.add_argument('-n_jobs', action='store', default=1, type=int, help='Number of parallel jobs')
parser.add_argument('-mem', action='store', dest='mem', default=1000, type=int, help='memory request and limit (MB)')
parser.add_argument('--slurm', action='store_true', default=False, help='Run on an slurm HPC')
parser.add_argument('-time', action='store', dest='time', default='4:00:00', type=str, help='time in HR:MN:SS')
args = parser.parse_args()

n_trials = len(args.SEEDS)
seeds = args.SEEDS.split(',')[:n_trials]
algs = args.algs.split(',')
Ss = args.Ss.split(',')
dims = args.Dims.split(',')
n_gens = args.n_gens.split(',')
diagnostic_ids = args.diagnostic_ids.split(',')
Ls = args.Ls.split(',')
damps = args.Damps.split(',')
eps = args.eps.split(',')
ep_types = args.ep_types.split(',')
args.slurm = True

# write run commands
all_commands = []
job_info = []
rdir = '/'.join([args.rdir])+algs
os.makedirs(rdir, exist_ok=True)

for alg,s,dim,damp,seed,epsilon,epsilon_type in it.product(algs,Ss,dims,damps,seeds,eps,ep_types):

	all_commands.append(
		f'python3 single_comparison_experiment.py -alg {algs} -S {s} -dim {dim} -damp {damp} -epsilon {epsilon} -epsilon_type {epsilon_type} -seed {seed} -rdir {rdir}'
	)

	job_info.append({
		'alg':alg,
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
	os.makedirs(rdir + 'jobfiles', exist_ok=True)
	for i, run_cmd in enumerate(all_commands):

		job_name = '_'.join([x + '-' + f'{job_info[i][x]}' for x in
							['alg','S','dim', 'damp', 'epsilon', 'epsilon_type', 'seed']])
		job_file = f'{rdir}/jobfiles/{job_name}.sb'
		out_file = job_info[i]['rdir'] + job_name + '_%J.out'

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
module purge
module restore new_modules
source lex/bin/activate
which python3
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
