import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.problems import get_problem

class DiagnosticProblem(ElementwiseProblem):

    def __init__(self, diagnostic_id, n_var, n_obj, xl, xu, **kwargs):

        self.diagnostic_id = diagnostic_id
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

        f = np.empty(self.n_obj)
        assert len(x) == len(f)
        total_score = 0
        pos = 0
        damp = self.damp

        match self.diagnostic_id:
            case 0:
                #EXPLOIT
                first_active = 0
                active_count = len(x)
                
                total_score = np.sum(f)
                out["F"] = -np.asarray(x)
                
            case 1:
                #STRUCT_EXPLOIT
                f[0] = x[0]
                first_active = 0
                for pos in range(len(x)):
                    if x[pos] <= x[pos - 1]:
                        f[pos] = x[pos]

                active_count = pos
                total_score = np.sum(f)
                out["F"] = -np.asarray(f)
                            
            case 2:
                #EXPLORE
                pos = np.argmax(x)
                f[pos] = x[pos]
                first_active = pos
                pos += 1

                while pos < len(x) and x[pos] <= x[pos - 1]:
                    f[pos] = x[pos]
                    pos += 1

                active_count = pos - first_active
                total_score = np.sum(f)
                out["F"] = -np.asarray(f)
                            
            case 3:     
                #DIVERSITY
                pos = np.argmax(x)
                f[pos] = x[pos]
                first_active = pos
                active_count = 1

                for i in range(len(x)):
                    if i != pos:
                        f[i] = (x[pos] - x[i]) / 2.0

                total_score = np.sum(f)
                out["F"] = -np.asarray(f)
            
            case 4:
                #WEAK_DIVERSITY
                pos = np.argmax(x)
                f[pos] = x[pos]
                first_active = pos
                active_count = 1

                total_score = np.sum(f)
                out["F"] = -np.asarray(f)
            
            case 5:
                #ANTAGONISTIC
                pos = np.argmax(x)
                first_active = pos
                active_count = 1

                f = (-1)*(x - np.sum(x)/damp + x/damp)

                total_score = np.sum(f)
                out["F"] = np.asarray(f)