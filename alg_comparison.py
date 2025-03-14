import numpy as np
import random
from diagnostics_problem import DiagnosticProblem
from diagnostics_problem import DiagnosticRandomSampling

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo_lexicase import Lexicase
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
#from algorithms import get_algorithm
from pymoo.problems import get_problem
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from algorithms.Lexicase import create_lexicase
from algorithms.NSGA2 import create_nsga2
from algorithms.NSGA3 import create_nsga3
from algorithms.MOEAD import create_moead

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

def get_ref_dirs(m):
    if m == 3:
        ref_dirs = get_reference_directions("das-dennis", m, n_partitions=12)
    elif m == 5:
        ref_dirs = get_reference_directions("das-dennis", m, n_partitions=6)
    elif m == 8:
        ref_dirs = get_reference_directions(
        "multi-layer",
        get_reference_directions("das-dennis", m, n_partitions=3, scaling=1.0),
        get_reference_directions("das-dennis", m, n_partitions=2, scaling=0.5))
    elif m == 10:
        ref_dirs = get_reference_directions(
        "multi-layer",
        get_reference_directions("das-dennis", m, n_partitions=3, scaling=1.0),
        get_reference_directions("das-dennis", m, n_partitions=2, scaling=0.5))
    elif m == 15:
        ref_dirs = get_reference_directions(
        "multi-layer",
        get_reference_directions("das-dennis", m, n_partitions=2, scaling=1.0),
        get_reference_directions("das-dennis", m, n_partitions=1, scaling=0.5))
    elif m > 15:
        ref_dirs = get_reference_directions(
        "multi-layer",
        get_reference_directions("das-dennis", m, n_partitions=1, scaling=1.0),
        get_reference_directions("das-dennis", m, n_partitions=1, scaling=0.75),
        get_reference_directions("das-dennis", m, n_partitions=1, scaling=0.5),
        get_reference_directions("das-dennis", m, n_partitions=1, scaling=0.25),
        get_reference_directions("das-dennis", m, n_partitions=1, scaling=0.1))
    else:
        raise ValueError("Reference direction for this number of objectives not defined.")
    
    return ref_dirs

##### Parameters
# Problem parameters
pop_size = 10
n_var = 5
n_obj = n_var
n_gen = 10
alg_name = "lex_std"
diagnostic = "antagonistic"
xl = 0
xu = 10
damp = 1

# Parameters for Lexicase
epsilon_type = 'constant'
epsilon = 0.0

# Parameters for NSGA3 and MOEA/D
#ref_dirs = get_reference_directions("das-dennis", n_var, n_partitions=10) # Get reference directions for NSGA3
#ref_dirs = get_reference_directions("energy", n_var, n_points=20100, seed=1)
n_neighbors = 20
prob_neighbor_mating = 0.7

##### Define the problem
problem = DiagnosticProblem(diagnostic=diagnostic, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, damp = damp)
#problem = get_problem("dtlz1")
# pf = get_problem("zdt1").pareto_front()

###### Define the algorithm

# algorithm = NSGA2(
#     pop_size=100,
#     sampling=FloatRandomSampling(),
#     crossover=SBX(prob=1.0, eta=3.0, vtype=float),
#     mutation=PM(eta=20),
#     eliminate_duplicates=True, 
#     ref_dirs=ref_dirs
# )

# algorithm = get_algorithm(alg, pop_size = pop_size, epsilon_type = epsilon_type, epsilon = epsilon,
#                            ref_dirs = ref_dirs, n_neighbors = n_neighbors, prob_neighbor_mating = prob_neighbor_mating)
if alg_name == "NSGA2":
    algorithm = create_nsga2(pop_size = pop_size)
elif alg_name == "lex_std":
    algorithm = create_lexicase(pop_size = pop_size, epsilon_type = 'standard', epsilon = 0.0)
elif alg_name == "lex_const":
    algorithm = create_lexicase(pop_size = pop_size, epsilon_type = 'constant', epsilon = 0.5)
elif alg_name == "lex_semi":
    algorithm = create_lexicase(pop_size = pop_size, epsilon_type = 'semi-dynamic', epsilon = None)
elif alg_name == "lex_dyn":
    algorithm = create_lexicase(pop_size = pop_size, epsilon_type = 'dynamic', epsilon = None)
elif alg_name == "NSGA3":
    ref_dirs = get_ref_dirs(n_obj)
    algorithm = create_nsga3(pop_size = ((len(ref_dirs) // 4) + 1) * 4, ref_dirs = ref_dirs)
elif alg_name == "MOEAD":
    ref_dirs = get_ref_dirs(n_obj)
    algorithm = create_moead(ref_dirs = ref_dirs, n_neighbors = n_neighbors, prob_neighbor_mating = prob_neighbor_mating)
else:
    raise ValueError("Invalid algorithm name / algorithm not implemented.")

###### Define the termination criteria
termination = get_termination("n_gen", n_gen)

###### Optimize the problem (maximizing f(x) is equal to minimizing -f(x))
res = minimize(problem,
               algorithm,
               termination,
               seed = 0,
               save_history=False,
               verbose=True)

##### Optimization results
# X_h = res.history[-1].pop.get('X') # Genotypes in final population
# F_h = res.history[-1].pop.get('F') # Phenotypes in final population
X = res.pop.get('X') # Genotypes in final population
F = res.pop.get('F') # Phenotypes in final population
opt_X = res.opt.get("X") # Optimal Genotypes (on the PF)
opt_F = res.opt.get("F") # Optimal Phenotypes (on the PF)

np.set_printoptions(precision=2, suppress=True)
print("X: ", X)
print("F: ", F)
print("opt_X: ", opt_X)
print("opt_F: ", opt_F)
print("pf_size: ", len(opt_F))

if diagnostic in ["exploit", "structExploit", "explore"]:
    score = np.min(np.mean(F, axis=1))
elif diagnostic in ["weakDiversity", "diversity", "antagonistic"]:
    score = np.sum(np.min(F, axis=0) < -9)

print("Final population score: ", score)

##### Hypervolume
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.hv.monte_carlo import ApproximateMonteCarloHypervolume
from pymoo.indicators.hv.exact import ExactHypervolume
from pymoo.indicators.hv.exact_2d import ExactHypervolume2D

ref_point = np.array([10]*problem.n_var)

if n_obj <= 3:
    ind = Hypervolume(pf = opt_F, ref_point=ref_point)
    hv = ind(opt_F)
else:
    ind = ApproximateMonteCarloHypervolume
    hv_ind = ind(F = opt_F, ref_point=ref_point)
    hv, hvc = hv_ind._calc(F = opt_F, ref_point = ref_point)

print('HV: ', hv)
##### Generational distance
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD

if diagnostic == "antagonistic":

    ref_pf_ints = ref_pf(problem, points_type = 'ints')
    ref_pf_corner = ref_pf(problem, points_type = 'corners')
    ref_pf_middle = ref_pf(problem, points_type = 'middles')
    ref_pf_zeros = np.array([[0]*n_obj])
    ref_pf_ = ref_pf_ints
    reference_pfs = ["corners", "middles", "zeros", "ints"]

    for p in reference_pfs:

        if p == "corners":
            indigd = IGD(pf = ref_pf_corner)
            igd = igd_corner = indigd(opt_F)
            indgd = GD(pf = ref_pf_corner)
            gd = gd_corner = indgd(opt_F)
            
        elif p == "middles":
            indigd = IGD(pf = ref_pf_middle)
            igd = igd_middle = indigd(opt_F)
            indgd = GD(pf = ref_pf_middle)
            gd = gd_middle = indgd(opt_F)
            
        elif p == "zeros":
            indigd = IGD(pf = ref_pf_zeros)
            igd = igd_zeros = indigd(opt_F)
            indgd = GD(pf = ref_pf_zeros)
            gd = gd_zeros = indgd(opt_F)

        elif p == "ints":
            indigd = IGD(pf = ref_pf_ints)
            igd = igd_ints = indigd(opt_F)
            indgd = GD(pf = ref_pf_ints)
            gd = gd_ints = indgd(opt_F)
            
        else:
            raise ValueError("Invalid reference pareto_front. Choose from 'corners', 'middles', 'zeros', 'ints'.")

        print(f"GD_{p}: {gd}")
        print(f"IGD_{p}: {igd}")

elif diagnostic in ['exploit', 'structExploit', 'explore']:
    ref_pf_ = np.array([-10]*n_obj)
    indigd = IGD(pf = ref_pf_)
    igd = indigd(opt_F)
    indgd = GD(pf = ref_pf_)   
    gd = indgd(opt_F)
    print(f"IGD: ", igd)
    print(f"GD: ", gd)

elif diagnostic == 'weakDiversity':
    arr = np.zeros((n_obj, n_obj))
    np.fill_diagonal(arr, -10)
    ref_pf_ = arr
    indigd = IGD(pf = ref_pf_)
    igd = indigd(opt_F)
    intgd = GD(pf = ref_pf_)
    gd = intgd(opt_F)
    print(f"IGD: ", igd)
    print(f"GD: ", gd)

elif diagnostic == 'diversity':
    arr = np.full((n_obj, n_obj), -5)
    np.fill_diagonal(arr, -10)
    ref_pf_ = arr
    indigd = IGD(pf = ref_pf_)
    igd = indigd(opt_F)
    indgd = GD(pf = ref_pf_)
    gd = indgd(opt_F)
    print(f"IGD: ", igd)
    print(f"GD: ", gd)

else:
    raise ValueError("Invalid diagnostic. Choose from 'exploit', 'structExploit', 'explore', 'diversity', 'weakDiversity', 'antagonistic'.")

##### Spacing Indicator
from pymoo.indicators.spacing import SpacingIndicator
indspace = SpacingIndicator()
if (len(opt_F)>1):
    spacing = indspace(opt_F)
else:
    spacing = -1
print("Spacing: ", spacing)

##### Plotting
# from pymoo.visualization.scatter import Scatter
# #plot = Scatter().add(F).save("nsga32d.png")
# from pymoo.visualization.star_coordinate import StarCoordinate
# from pymoo.indicators.hv import Hypervolume
# # StarCoordinate().add(opt_F).save("star_pareto")
# # StarCoordinate().add(true_pf).save("star_true_pareto")
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting if needed

# #Ensure the two sets of points are numpy arrays
# opt_F = np.array(opt_F)
# true_pf = np.array(ref_pf_)

# if opt_F.shape[1] == 2:  # 2D plotting
#     plt.figure(figsize=(8, 6))
#     plt.scatter(opt_F[:, 0], opt_F[:, 1], color='red', label='opt_F', alpha=0.7)
#     plt.scatter(true_pf[:, 0], true_pf[:, 1], color='blue', label='pareto_front', alpha=0.7)
#     #x_line, y_line = plot_line_from_normal_vector(normal_vector)
#     #plt.plot(x_line, y_line, color='blue', label='Line from normal vector', linewidth=2)
#     plt.xlabel('Objective 1')
#     plt.ylabel('Objective 2')
#     plt.title(f'plot of solutions vs true pareto front (2D)\npop_size: {pop_size}, n_generations: {n_gen}\nHV: {hv:.3f}, GD: {gd:.3f}')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f'/home/shakiba/MultiObjectivePrediction/plots/standard_lexicase_2D_damp2.png')

# elif opt_F.shape[1] == 3:  # 3D plotting
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(opt_F[:, 0], opt_F[:, 1], opt_F[:, 2], color='red', label='opt_F', alpha=0.7)
#     ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], color='blue', label='pareto_front', alpha=0.7)
#     ax.set_xlabel('Objective 1')
#     ax.set_ylabel('Objective 2')
#     ax.set_zlabel('Objective 3')
#     ax.set_title(f'plot of solutions vs true pareto front (3D)\npop_size: {pop_size}, n_generations: {n_gen}\nHV: {hv:.3f}, GD: {gd:.3f}')
#     ax.legend()
#     plt.grid()
#     plt.savefig(f'/home/shakiba/MultiObjectivePrediction/plots/{alg_name}_pfs_3D_test.png')

# else:
#     print("Cannot plot data with dimensions higher than 3.")

# import plotly.graph_objects as go

# fig = go.Figure()
# if diagnostic == "antagonistic":
#     blue_points = ref_pf_ints
# else:
#     blue_points_x = np.array([[i, 0, 0] for i in range(0, 11)])  # Points on x-axis
#     blue_points_y = np.array([[0, i, 0] for i in range(0, 11)])  # Points on y-axis
#     blue_points_z = np.array([[0, 0, i] for i in range(0, 11)])  # Points on z-axis
#     blue_points = np.vstack([blue_points_x, blue_points_y, blue_points_z])

# fig.add_trace(go.Scatter3d(
#     x=opt_F[:, 0], y=opt_F[:, 1], z=opt_F[:, 2],
#     mode='markers',
#     marker=dict(size=5, color='red', opacity=0.7),
#     name='opt_F'
# ))

# fig.add_trace(go.Scatter3d(
#     x=blue_points[:, 0], y=blue_points[:, 1], z=blue_points[:, 2],
#     mode='markers',
#     marker=dict(size=5, color='blue', opacity=0.4),
#     name='pareto_front'
# ))

# fig.update_layout(
#     scene=dict(
#         xaxis_title='Objective 1',
#         yaxis_title='Objective 2',
#         zaxis_title='Objective 3',
#     ),
#     title=f'{alg_name}, {diagnostic} <br>Solutions (red) vs True Pareto Front (blue) <br>pop_size: {pop_size}, n_generations: {n_gen},<br>HV: {hv:.3f}, IGD: {igd:.3f}, spacing: {spacing:.3f}',
#     legend=dict(x=0.8, y=0.9)
# )

# #Save to file or show
# #fig.write_html(f'/home/shakiba/MultiObjectivePrediction/plots/{alg_name}_{diagnostic}_3D.html')
# fig.show()
