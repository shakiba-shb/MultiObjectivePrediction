import numpy as np
import pandas as pd

import random
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.survival import Survival
from pymoo.core.selection import Selection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.misc import has_feasible
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.selection.rnd import RandomSelection

def get_parent(pop, epsilon_type, epsilon):

    phenotypes = pop.get("F")

    G = np.arange(len(phenotypes[0]))
    S = np.arange(len(pop))
    fitness = []

    if (epsilon_type == 'semi-dynamic'):
        ep = np.zeros(len(G))

        for i in range(len(G)):
            ep[i] = np.median(np.abs(phenotypes[:, i] - np.median(phenotypes[:, i])))

    while (len(G) > 0 and len(S) > 1):

        g = random.choice(G)
        fitness = phenotypes[:, g]
        L = min(fitness) 

        if (epsilon_type == 'dynamic'):
            if not hasattr(get_parent, "_called"):
                print("dynamic epsilon called")
                get_parent._called = True
            epsilon = np.median(np.abs(fitness - np.median(fitness)))

        elif (epsilon_type == 'semi-dynamic'):
            if not hasattr(get_parent, "_called"):
                print("semi-dynamic epsilon called")
                get_parent._called = True
            epsilon = ep[g]

        elif (epsilon_type == 'constant'):
            if not hasattr(get_parent, "_called"):
                print("constant epsilon called")
                get_parent._called = True
            epsilon = epsilon

        elif (epsilon_type == 'standard'):
            if not hasattr(get_parent, "_called"):
                print("standard epsilon called")
                get_parent._called = True
            epsilon = 0.0

        else:
            raise ValueError('Invalid epsilon type')

        survivors = np.where(fitness <= L + epsilon)
        S = S[survivors]
        G = G[np.where(G != g)]
        phenotypes = phenotypes[survivors]
            
    S = S[:, None].astype(int, copy=False)     
    return random.choice(S)

class LexicaseSelection(Selection):
    
    def __init__(self, epsilon_type, epsilon, **kwargs):
        super().__init__(**kwargs)

        self.epsilon_type = epsilon_type
        self.epsilon = epsilon
     
    def _do(self, _, pop, n_select, n_parents=1, **kwargs):

        parents = []

        for i in range(n_select * n_parents): 
            #get pop_size parents
            p = get_parent(pop, self.epsilon_type, self.epsilon)
            parents.append(p)

        return np.reshape(parents, (n_select, n_parents))


class LexSurvival(Survival):
    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        return pop[-n_survive:]

class Lexicase(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=LexicaseSelection(epsilon_type='constant', epsilon=0),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=LexSurvival(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)
        
        self.termination = DefaultMultiObjectiveTermination()
