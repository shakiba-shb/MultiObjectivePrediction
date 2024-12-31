import uuid
import numpy as np
import json
import argparse
import os
import random

from diagnostics_problem import DiagnosticProblem
from algorithms import get_algorithm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo_lexicase import Lexicase
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.gd import GD
from pymoo.indicators.spacing import SpacingIndicator

def sample_true_pf(D, L, sample_size):
    """
    Generate a set of pop_size points that lie on the true pareto front.
    
    Parameters:
    D (int): The number of dimensions.
    L (int): The magnitude of the coordinates.
    sample_size (int): The number of points to generate.
    
    Returns:
    np.ndarray: The generated points.
    """
    sample_pf = set() # we use set to make sure points are unique
    while len(sample_pf) < sample_size:
        I = random.randrange(0, D)
        value = round(random.uniform(0, L), 2)
        point = tuple([-value if j == I else value for j in range(D)]) # make sure it's negative since we are minimizing
        sample_pf.add(point)
    sample_pf.add(tuple([0]*D)) # Add solution with all 0s
    return np.array([np.array(point) for point in sample_pf])

def experiment (S = None, dim = None, n_gen = None, diagnostic_id = None, L = None, 
                damp = None, epsilon = None, epsilon_type = None, seed = None, rdir = ""):
    
    runid = uuid.uuid4()
    for alg in ["Lexicase", "NSGA2"]:

        print("alg = ", alg, "S = ", S, "n_var = ", dim, "n_obj = ", dim, "L = ", L, "n_gen = ", n_gen, "damp = ", damp,
            "epsilon = ", epsilon, "epsilon_type = ", epsilon_type, "seed = ", seed, "rdir = ", rdir, "runid = ", runid)
        
        #Define the problem
        problem = DiagnosticProblem(diagnostic_id=diagnostic_id, n_var=dim, n_obj=dim, xl=0, xu=L, damp = 1)

        #Define the algorithms
        #all algorithms should be run in the same experiment to be compared with the same true pf
        algorithm = get_algorithm(alg, pop_size = S)

        #Define the termination criteria
        termination = get_termination("n_gen", n_gen)

        #Optimize the problem (maximizing f(x) is equal to minimizing -f(x))
        res = minimize(problem,
                    algorithm,
                    termination,
                    seed = seed,
                    save_history=True,
                    verbose=True)
        
        np.set_printoptions(precision=2, suppress=True)
        X = res.history[-1].pop.get('X') #final genotypes
        F = res.history[-1].pop.get('F') #final phenotypes
        opt_X = res.opt.get("X") #final solutions genotypes (pf)
        opt_F = res.opt.get("F") #final solutions phenotypes (pf)

        #Hypervolume calculation
        # ref_point = np.array([L]*problem.n_var)
        # ind = Hypervolume(pf = opt_F, ref_point=ref_point)
        # hv = ind._do(opt_F)
        #hv_norm = hv / np.prod(ref_point)

        true_pf = sample_true_pf(D = dim, L = L, sample_size = S)
        indgd = GD(pf = true_pf)
        gd = indgd(opt_F)
        print("GD: ", gd)

        indspace = SpacingIndicator()
        spacing = indspace(opt_F)

        result = {'alg': alg, 'S': S, 'dim':dim, 'n_gen':n_gen, 'diagnostic_id':diagnostic_id, 'L':L,'GD':float(gd),'spacing':float(spacing), 'damp':damp, 'epsilon':epsilon, 
                'epsilon_type':epsilon_type, 'seed':seed, 'rdir':rdir}
        print(result)
        final_population_data = {'X': X.tolist(), 'F': F.tolist(), 'opt_X': opt_X.tolist(), 'opt_F': opt_F.tolist()}

        filename = rdir + f'{alg}/runid-{runid}.json'
        os.makedirs(rdir+f'{alg}', exist_ok=True)
        with open(filename, 'w') as of:
            json.dump(result, of, indent=2)

        if (seed == 14724):
            os.makedirs(rdir+'experiment', exist_ok=True)
            filename_pop = rdir + f'/experiment/alg-{alg}-S-{S}-dim-{dim}-seed-{seed}.json'
            with open(filename_pop, 'w') as f:
                json.dump(final_population_data, f, indent=2)
        
        
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a single comparison experiment')
    #parser.add_argument('-alg', type=str, default = 'Lexicase', help='Algorithm to use')
    parser.add_argument('-S', type=int, default = 100, help='Population size')
    parser.add_argument('-dim', type=int, default = 5, help='Number of objectives/variables')
    parser.add_argument('-n_gen', type=int, default = 50, help='Number of generations')
    parser.add_argument('-diagnostic_id', type=int, default = 5, help='Diagnostic problem id')
    parser.add_argument('-L', type=int, default = 10, help='Search space limit')
    parser.add_argument('-damp', type=float, default = 1.0, help='Dampening factor')
    parser.add_argument('-epsilon', type=float, default = 0.0, help='Epsilon value')
    parser.add_argument('-epsilon_type', type=int, default = 0, help='Epsilon type')
    parser.add_argument('-seed', type=int, default = 14724, help='Random seed')
    parser.add_argument('-rdir', type=str, default = '/home/shakiba/MultiObjectivePrediction/results/', help='Results directory')
    args = parser.parse_args()

    experiment(S = args.S, dim = args.dim, n_gen = args.n_gen, diagnostic_id = args.diagnostic_id, L = args.L,
                damp = args.damp, epsilon = args.epsilon, epsilon_type = args.epsilon_type, seed = args.seed, rdir = args.rdir)

