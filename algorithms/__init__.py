
from .Lexicase import create_lexicase
from .NSGA2 import create_nsga2
#from NSGA3 import create_nsga3

def get_algorithm(alg_name, pop_size, **kwargs):
    algorithms = {
        "Lexicase": create_lexicase(pop_size = pop_size, epsilon_type = kwargs.get("epsilon_type"), epsilon = kwargs.get("epsilon")),
        "NSGA2": create_nsga2(pop_size = pop_size),
        # "NSGA3": create_nsga3(pop_size = pop_size, ref_dirs = kwargs.get("ref_dirs")
    }
    if alg_name not in algorithms:
        raise ValueError(f"Unsupported algorithm: {alg_name}. Supported algorithms: {list(algorithms.keys())}")
    return algorithms[alg_name]