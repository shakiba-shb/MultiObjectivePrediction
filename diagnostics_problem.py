import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.problems import get_problem
from pymoo.core.sampling import Sampling

class DiagnosticProblem(ElementwiseProblem):

    def __init__(self, diagnostic, n_var, n_obj, xl, xu, **kwargs):

        self.diagnostic = diagnostic
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu
        self.damp = kwargs["damp"]

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         xl=np.array([self.xl] * self.n_var),
                         xu=np.array([self.xu] * self.n_var),
                         **kwargs
                         )

    def _evaluate(self, x, out, *args, **kwargs):

        f = np.zeros(self.n_obj)
        assert len(x) == len(f)
        total_score = 0
        pos = 0
        damp = self.damp

        match self.diagnostic:
            case 'exploit':
                # EXPLOITATION RATE  
                first_active = 0
                active_count = len(x)
                
                total_score = -np.sum(f)
                out["F"] = -np.asarray(x)
                
            case 'structExploit':
                # ORDERED EXPLOITATION RATE
                f[0] = x[0]
                first_active = 0
                for pos in range(1, len(x)):
                    if x[pos] <= x[pos - 1]:
                        f[pos] = x[pos]
                    else:
                        f[pos:] = 0
                        break

                active_count = pos
                total_score = -np.sum(f)
                out["F"] = -np.asarray(f)
                            
            case 'explore':
                # EXPLORATION RATE
                pos = np.argmax(x)
                f[pos] = x[pos]
                first_active = pos
                pos += 1

                while pos < len(x) and x[pos] <= x[pos - 1]:
                    f[pos] = x[pos]
                    pos += 1

                active_count = pos - first_active
                total_score = -np.sum(f)
                out["F"] = -np.asarray(f)
                            
            case 'diversity':     
                # DIVERSITY
                pos = np.argmax(x)
                f[pos] = x[pos]
                first_active = pos
                active_count = 1

                for i in range(len(x)):
                    if i != pos:
                        f[i] = (x[pos] - x[i]) / 2.0

                total_score = -np.sum(f)
                out["F"] = -np.asarray(f)
            
            case 'weakDiversity':
                # CONTRADICTORY OBJECTIVES
                pos = np.argmax(x)
                f[pos] = x[pos]
                first_active = pos
                active_count = 1

                total_score = -np.sum(f)
                out["F"] = -np.asarray(f)
            
            case 'antagonistic':
                # ANTAGONISTIC CONTRADICTORY OBJECTIVES (WORST CASE)
                pos = np.argmax(x)
                first_active = pos
                active_count = 1

                f = (x - np.sum(x)/damp + x/damp)

                total_score = np.sum(f)
                out["F"] = -np.asarray(f)


class DiagnosticRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.random((n_samples, problem.n_var))

        return X