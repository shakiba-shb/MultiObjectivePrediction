from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.sampling.rnd import FloatRandomSampling
from diagnostics_problem import DiagnosticRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

def create_moead(pop_size, ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7):
    return MOEAD(
        #pop_size=pop_size,
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        prob_neighbor_mating=prob_neighbor_mating,
        sampling=DiagnosticRandomSampling(),
        crossover=SBX(prob=1.0, eta=3.0, vtype=float),
        mutation=PM(eta=20),
        #eliminate_duplicates=True
    )