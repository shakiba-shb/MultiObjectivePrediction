import numpy as np
import random
from diagnostics_problem import DiagnosticProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo_lexicase import Lexicase
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from algorithms import get_algorithm
from pymoo.problems import get_problem

def find_hyperplane(*points):
    """
    Finds the hyperplane that includes n points in n-dimensional space and returns its normal vector and d.

    Parameters:
        points (tuple): A variable number of points, each as a tuple (x1, x2, ..., xn) in n-dimensional space.

    Returns:
        tuple: A tuple containing the normal vector and d.
    """
    points = np.array(points)
    n_points, n_dims = points.shape

    # Ensure the correct number of points for the dimension
    if n_points != n_dims:
        raise ValueError("The number of points must match the dimensionality of the space.")

    # Form a matrix with point differences
    diff_matrix = points[1:] - points[0]

    # Compute the null space (normal vector to the hyperplane)
    u, s, vh = np.linalg.svd(diff_matrix)
    normal_vector = vh[-1]  # Null space corresponds to the last row of Vh in SVD

    # Ensure the normal vector is not zero
    if np.allclose(normal_vector, 0):
        raise ValueError("The points do not define a unique hyperplane.")

    # Calculate d using the hyperplane equation
    d = -np.dot(normal_vector, points[0])

    return normal_vector, d


def find_true_pf(normal_vector, num_points=100):
    # Number of dimensions
    n = len(normal_vector)
    x_range=(-10, 10*(n-1))
    
    # Randomly sample values for x1, x2, ..., x_(n-1), because we can't use the whole pareto front
    sampled_points = np.random.uniform(x_range[0], x_range[1], (num_points, n - 1))
    
    # Calculate the corresponding xn values based on the line equation in n dimensions
    # x_n = -(a_1/a_n) * x_1 - (a_2/a_n) * x_2 - ... - (a_(n-1)/a_n) * x_(n-1)
    x_n_values = -np.sum(sampled_points * normal_vector[:-1], axis=1) / normal_vector[-1]
    
    # Append the xn values to form the full n-dimensional points
    pareto_front = np.column_stack((sampled_points, x_n_values))
    
    return pareto_front

def generate_points(L, D):
    """
    Generate points where each position is L once, while every other position is -L.
    These points, along with the point where are positions are 0, are used to generate the true pareto front.
    
    Parameters:
    N (int): The magnitude of the coordinates.
    D (int): The number of dimensions.
    
    Returns:
    list of tuples: The generated points.
    """
    points = []
    for i in range(D):
        # Create a point where the i-th position is N and all others are -N
        point = [-L] * D
        point[i] = L
        points.append(tuple(point))
    return points

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

##### Parameters
pop_size = 100
n_var = 3
n_obj = n_var
n_gen = 100
alg = "Lexicase"
xl = 0
xu = 10
damp = 1

##### Define the problem
problem = DiagnosticProblem(diagnostic_id=5, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, damp = damp)
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

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12) # Get reference directions for NSGA3
algorithm = get_algorithm(alg, pop_size = pop_size)

###### Define the termination criteria
termination = get_termination("n_gen", n_gen)

###### Optimize the problem (maximizing f(x) is equal to minimizing -f(x))
res = minimize(problem,
               algorithm,
               termination,
               seed = 0,
               save_history=True,
               verbose=True)

##### Optimization results
X = res.history[-1].pop.get('X') # Genotypes in final population
F = res.history[-1].pop.get('F') # Phenotypes in final population
opt_X = res.opt.get("X") # Optimal Genotypes (on the PF)
opt_F = res.opt.get("F") # Optimal Phenotypes (on the PF)

np.set_printoptions(precision=2, suppress=True)
print("X: ", X)
print("F: ", F)
print("opt_X: ", opt_X)
print("opt_F: ", opt_F)
print("pf_size: ", len(opt_F))


##### Hypervolume
from pymoo.indicators.hv import Hypervolume
# ref_point = np.array([10*(problem.n_var - 1)]*problem.n_var) # Get reference point for hypervolume calculation
ref_point = np.array([10]*problem.n_var)
# #print(ref_point)
ind = Hypervolume(pf = opt_F, ref_point=ref_point)
hv = ind(opt_F)
print("HV: ", hv)

##### Generational distance
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD

# 2d example
# if opt_F.shape[1] == 2:
#     p1 = (0, 0)
#     p2 = (10, -10)
#     p3 = (-10, 10)
#     normal_vector, d = find_hyperplane(p1, p2, p3) # Calculate the normal vector and d value of hyperplane including points p(i)

# 3D example
# if opt_F.shape[1] == 3:
#     p1 = (0, 0, 0)
#     p2 = (-10, -10, 10)
#     p3 = (10, -10, -10)
#     p4 = (-10, 10, -10)
#     normal_vector, d = find_hyperplane2(p2, p3, p4) # Calculate the normal vector and d value of hyperplane including points p(i)

# p1 = tuple([0]*n_var)
# points = generate_points(xu, n_var)    
# normal_vector, d = find_hyperplane(p1, *points)

# true_pf = find_true_pf(normal_vector = normal_vector, num_points = pop_size) # Generate Pareto front from normal vector
# #true_pf = problem.pareto_front
    
true_pf = sample_true_pf(n_obj, xu, pop_size)
indgd = GD(pf = true_pf)
indigd = IGD(pf = true_pf)

gd = indgd(opt_F)
igd = indigd(opt_F)

print("GD: ", gd)
print("IGD: ", igd)

##### Spacing Indicator
from pymoo.indicators.spacing import SpacingIndicator
indspace = SpacingIndicator()
spacing = indspace(opt_F)
print("Spacing: ", spacing)

##### Plotting
from pymoo.visualization.scatter import Scatter
#plot = Scatter().add(F).save("nsga32d.png")
from pymoo.visualization.star_coordinate import StarCoordinate
from pymoo.indicators.hv import Hypervolume
# StarCoordinate().add(opt_F).save("star_pareto")
# StarCoordinate().add(true_pf).save("star_true_pareto")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting if needed

# Ensure the two sets of points are numpy arrays
# opt_F = np.array(opt_F)
# true_pf = np.array(true_pf)

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
#     plt.savefig(f'/home/shakiba/MultiObjectivePrediction/plots/{alg}_pfs_2D.png')

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
#     plt.savefig(f'/home/shakiba/MultiObjectivePrediction/plots/{alg}_pfs_3D_test.png')

# else:
#     print("Cannot plot data with dimensions higher than 3.")

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=opt_F[:, 0], y=opt_F[:, 1], z=opt_F[:, 2],
    mode='markers',
    marker=dict(size=5, color='red', opacity=0.7),
    name='opt_F'
))

fig.add_trace(go.Scatter3d(
    x=true_pf[:, 0], y=true_pf[:, 1], z=true_pf[:, 2],
    mode='markers',
    marker=dict(size=5, color='blue', opacity=0.7),
    name='pareto_front'
))

fig.update_layout(
    scene=dict(
        xaxis_title='Objective 1',
        yaxis_title='Objective 2',
        zaxis_title='Objective 3',
    ),
    title=f'{alg}<br>Solutions (red) vs True Pareto Front (blue) <br>pop_size: {pop_size}, n_generations: {n_gen},<br>HV: {hv:.3f}, GD: {gd:.3f}, IGD: {igd:.3f}, spacing: {spacing:.3f}',
    legend=dict(x=0.8, y=0.9)
)

# Save to file or show
#fig.write_html(f'/home/shakiba/MultiObjectivePrediction/plots/{alg}_pfs_3D_interactive_sample_pf.html')
fig.show()