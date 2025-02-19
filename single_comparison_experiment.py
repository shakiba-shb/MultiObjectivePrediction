import uuid
import numpy as np
import json
import argparse
import os
import random

from diagnostics_problem import DiagnosticProblem
#from algorithms import get_algorithm
from algorithms.Lexicase import create_lexicase
from algorithms.NSGA2 import create_nsga2
from algorithms.NSGA3 import create_nsga3
from algorithms.MOEAD import create_moead

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
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator

def ref_pf(problem, points_type):
    """
    Generate a reference Pareto front containing points based on the specified type.
    
    Parameters:
    - problem: The problem instance with attributes `n_obj` (number of objectives/problem dimension)
      and `xu` (upper bounds for variables).
    - points_type: Type of points to generate. Can be one of 'ints', 'corners', or 'middles'.
    
    Returns:
    - np.ndarray: Array of points on the Pareto front.
    """
    _points = set()  
    D = problem.n_obj 
    L = int(max(problem.xu))  

    if points_type == 'ints':
        # Generate all points where one value is an integer and the rest are 0
        for i in range(D):
            for j in range(L+1):
                genotype = np.zeros(D, dtype=int)
                genotype[i] = j
                problem._evaluate(genotype, out := {})
                _points.add(tuple(out["F"]))

    elif points_type == 'corners':
        # Generate corner points
        for i in range(D):
            genotype = np.zeros(D, dtype=int)
            genotype[i] = L
            problem._evaluate(genotype, out := {})
            _points.add(tuple(out["F"]))

    elif points_type == 'middles':
        # Generate middle points
        for i in range(D):
            genotype = np.zeros(D, dtype=int)
            genotype[i] = L/2
            problem._evaluate(genotype, out := {})
            _points.add(tuple(out["F"]))

    return np.array(list(_points))

def experiment (alg_name = None, S = None, dim = None, n_gen = None, diagnostic = None, L = None, damp = None, seed = None, rdir = ""):
    
    runid = uuid.uuid4()

    print("alg_name = ", alg_name, "S = ", S, "n_var = ", dim, "n_obj = ", dim, 'diagnostic = ', diagnostic, "L = ", L, "n_gen = ", n_gen, "damp = ", damp,
         "seed = ", seed, "rdir = ", rdir, "runid = ", runid)
    
    #Define the problem
    problem = DiagnosticProblem(diagnostic=diagnostic, n_var=dim, n_obj=dim, xl=0, xu=L, damp = damp)

    #Define the algorithms
    
    if alg_name == "NSGA2":
        algorithm = create_nsga2(pop_size = S)
    elif alg_name == "lex_std":
        algorithm = create_lexicase(pop_size = S, epsilon_type = 'standard', epsilon = 0.0)
    elif alg_name == "lex_const":
        algorithm = create_lexicase(pop_size = S, epsilon_type = 'constant', epsilon = 0.5)
    elif alg_name == "lex_semi":
        algorithm = create_lexicase(pop_size = S, epsilon_type = 'semi-dynamic', epsilon = None)
    elif alg_name == "lex_dyn":
        algorithm = create_lexicase(pop_size = S, epsilon_type = 'dynamic', epsilon = None)
    elif alg_name == "NSGA3":
        ref_dirs = get_reference_directions("das-dennis", dim, n_partitions = 12)
        algorithm = create_nsga3(pop_size = S, ref_dirs = ref_dirs)
    elif alg_name == "MOEAD":
        ref_dirs = get_reference_directions("das-dennis", dim, n_partitions = 12)
        algorithm = create_moead(pop_size = S, ref_dirs = ref_dirs, n_neighbors = 15, prob_neighbor_mating = 0.7)
    else:
        raise ValueError("Invalid algorithm name / algorithm not implemented.")

    #Define the termination criteria
    termination = get_termination("n_gen", n_gen)

    #Optimize the problem (maximizing f(x) is equal to minimizing -f(x))
    res = minimize(problem,
                algorithm,
                termination,
                seed = seed,
                save_history=False,
                verbose=True)
    
    np.set_printoptions(precision=2, suppress=True)
    # X = res.history[-1].pop.get('X') #final genotypes
    # F = res.history[-1].pop.get('F') #final phenotypes
    X = res.pop.get('X') #final genotypes
    F = res.pop.get('F') #final phenotypes
    opt_X = res.opt.get("X") #final solutions genotypes (pf)
    opt_F = res.opt.get("F") #final solutions phenotypes (pf)

    assert len(X) == len(F), "X and F should have the same length"
    assert len(opt_X) == len(opt_F), "opt_X and opt_F should have the same length"

    ###### Performance Indicators ######

    #Hypervolume
    # ref_point = np.array([L]*problem.n_var)
    # ind = Hypervolume(pf = opt_F, ref_point=ref_point)
    # hv = ind._do(opt_F)
    #hv_norm = hv / np.prod(ref_point)

    #Inverted Generational Distance
    #true_pf = sample_true_pf(D = dim, L = L, sample_size = S)
    igd = float('inf')
    gd = float('inf')
    igd_corner = igd_middle = igd_zeros = igd_ints = float('inf')
    gd_corner = gd_middle = gd_zeros = gd_ints = float('inf')

    if diagnostic == "antagonistic":

        ref_pf_ints = ref_pf(problem, points_type = 'ints')
        ref_pf_corner = ref_pf(problem, points_type = 'corners')
        ref_pf_middle = ref_pf(problem, points_type = 'middles')
        ref_pf_zeros = np.array([[0]*dim])
        ref_pf_ = ref_pf_ints
        reference_pfs = ["corners", "middles", "zeros", "ints"]

        for p in reference_pfs:

            if p == "corners":
                indigd = IGD(pf = ref_pf_corner)
                igd_corner = indigd(opt_F)
                indgd = GD(pf = ref_pf_corner)
                gd_corner = indgd(opt_F)
                
            elif p == "middles":
                indigd = IGD(pf = ref_pf_middle)
                igd_middle = indigd(opt_F)
                indgd = GD(pf = ref_pf_middle)
                gd_middle = indgd(opt_F)
                
            elif p == "zeros":
                indigd = IGD(pf = ref_pf_zeros)
                igd_zeros = indigd(opt_F)
                indgd = GD(pf = ref_pf_zeros)
                gd_zeros = indgd(opt_F)

            elif p == "ints":
                indigd = IGD(pf = ref_pf_ints)
                igd_ints = indigd(opt_F)
                indgd = GD(pf = ref_pf_ints)
                gd_ints = indgd(opt_F)
                
            else:
                raise ValueError("Invalid reference pareto_front. Choose from 'corners', 'middles', 'zeros', 'ints'.")
            

    elif diagnostic in ['exploit', 'structExploit', 'explore']:
        ref_pf_ = np.array([-10]*dim)
        indigd = IGD(pf = ref_pf_)
        igd = indigd(opt_F)

    elif diagnostic == 'weakDiversity':
        arr = np.zeros((dim, dim))
        np.fill_diagonal(arr, -10)
        ref_pf_ = arr
        indigd = IGD(pf = ref_pf_)
        igd = indigd(opt_F)
        indgd = GD(pf = ref_pf_)
        gd = indgd(opt_F)

    elif diagnostic == 'diversity':
        arr = np.full((dim, dim), -5)
        np.fill_diagonal(arr, -10)
        ref_pf_ = arr
        indigd = IGD(pf = ref_pf_)
        igd = indigd(opt_F)
        indgd = GD(pf = ref_pf_)
        gd = indgd(opt_F)

    else:
        raise ValueError("Invalid diagnostic. Choose from 'exploit', 'structExploit', 'explore', 'diversity', 'weakDiversity', 'antagonistic'.")

    #Spacing
    indspace = SpacingIndicator()
    if (len(opt_F) > 1): 
        spacing = indspace(opt_F)
    else:
        spacing = -1 # spacing is not defined if there's only one soltuion on the pf

    result = {'alg_name': alg_name, 'S': S, 'dim':dim, 'n_gen':n_gen, 'diagnostic':diagnostic, 'L':L,
                'IGD':float(igd), 'IGD_corners':float(igd_corner),'IGD_middles':float(igd_middle),'IGD_zeros':float(igd_zeros),'IGD_ints':float(igd_ints),  
                'GD':float(gd), 'GD_corners':float(gd_corner),'GD_middles':float(gd_middle),'GD_zeros':float(gd_zeros),'GD_ints':float(gd_ints),
                'spacing':float(spacing), 'pf_size':len(opt_F), 'damp':damp, 'seed':seed, 'rdir':rdir}
    
    print(result)
    final_population_data = {'X': X.tolist(), 'F': F.tolist(), 'opt_X': opt_X.tolist(), 'opt_F': opt_F.tolist()}

    folder_name = rdir+f'{alg_name}'
    filename = folder_name+f'/runid-{runid}.json'
    os.makedirs(folder_name, exist_ok=True)
    with open(filename, 'w') as of:
        json.dump(result, of, indent=2)

    os.makedirs(folder_name+'/experiment', exist_ok=True)
    filename_pop = folder_name + f'/experiment/alg-{alg_name}-S-{S}-dim-{dim}-seed-{seed}.json'
    with open(filename_pop, 'w') as f:
        json.dump(final_population_data, f, indent=2)
    
    
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a single comparison experiment')
    parser.add_argument('-alg_name', type=str, default = 'NSGA2', help='Algorithm to use')
    parser.add_argument('-S', type=int, default = 100, help='Population size')
    parser.add_argument('-dim', type=int, default = 3, help='Number of objectives/variables')
    parser.add_argument('-n_gen', type=int, default = 100, help='Number of generations')
    parser.add_argument('-diagnostic', type=str, default = 'exploit', help='Diagnostic problem')
    parser.add_argument('-L', type=int, default = 10, help='Search space limit')
    parser.add_argument('-damp', type=float, default = 1.0, help='Dampening factor')
    #parser.add_argument('-epsilon', type=float, default = '0.0', help='Epsilon value')
    #parser.add_argument('-epsilon_type', type=str, default = 'constant', help='Epsilon type')
    parser.add_argument('-seed', type=int, default = 0, help='Random seed')
    parser.add_argument('-rdir', type=str, default = '/home/shakiba/MultiObjectivePrediction/results/', help='Results directory')
    args = parser.parse_args()

    experiment(alg_name = args.alg_name, S = args.S, dim = args.dim, n_gen = args.n_gen, diagnostic = args.diagnostic, L = args.L,
                damp = args.damp, seed = args.seed, rdir = args.rdir)

