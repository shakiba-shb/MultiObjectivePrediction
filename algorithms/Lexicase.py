from pymoo_lexicase import Lexicase, LexicaseSelection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from diagnostics_problem import DiagnosticRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

def create_lexicase(pop_size, epsilon_type = 'constant', epsilon=0):
    return Lexicase(
        pop_size=pop_size,
        selection=LexicaseSelection(epsilon_type=epsilon_type, epsilon=epsilon),
        sampling=DiagnosticRandomSampling(),
        crossover=SBX(prob=1.0, eta=3.0, vtype=float),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )