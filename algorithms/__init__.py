
from .Lexicase import create_lexicase
from .NSGA2 import create_nsga2
#from NSGA3 import create_nsga3

def get_algorithm(alg_name, pop_size):
    algorithms = {
        "Lexicase": create_lexicase,
         "NSGA2": create_nsga2,
        # "NSGA3": create_nsga3
    }
    if alg_name not in algorithms:
        raise ValueError(f"Unsupported algorithm: {alg_name}. Supported algorithms: {list(algorithms.keys())}")
    return algorithms[alg_name](pop_size)