from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from diagnostics_problem import DiagnosticRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

def create_nsga2(pop_size):
    return NSGA2(
        pop_size=pop_size,
        sampling=DiagnosticRandomSampling(),
        crossover=SBX(prob=1.0, eta=3.0, vtype=float),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )