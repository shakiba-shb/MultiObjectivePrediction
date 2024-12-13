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

def get_parent(pop):
    epsilon = 0
    phenotypes = pop.get("F")
    
    G = np.arange(len(phenotypes[0]))
    S = np.arange(len(pop))
    fitness = []

    while (len(G) > 0 and len(S) > 1):

        g = random.choice(G)
        fitness = phenotypes[:, g]
        L = min(fitness) 

        survivors = np.where(fitness == L + epsilon)
        S = S[survivors]
        G = G[np.where(G != g)]
        phenotypes = phenotypes[survivors]
            
    S = S[:, None].astype(int, copy=False)     
    return random.choice(S)

class FLEX(Selection):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
     
         
    def _do(self, _, pop, n_select, n_parents=1, **kwargs):

        parents = []

        for i in range(n_select * n_parents): 
            #get pop_size parents
            p = get_parent(pop)
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
                 epsilon_type = None,
                 sampling=FloatRandomSampling(),
                 selection=FLEX(),
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
        self.epsilon_type = epsilon_type