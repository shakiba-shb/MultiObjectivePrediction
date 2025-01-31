import pytest
from diagnostics_problem import DiagnosticProblem
import numpy as np
    
@pytest.mark.parametrize("diagnostic, geno_pheno_pairs", [
        ("exploit", [
            (np.array([0, 0, 0, 0, 0]), -np.array([0, 0, 0, 0, 0])),
            (np.array([10, 0, 10, 0, 0]), -np.array([10, 0, 10, 0, 0])),
            (np.array([0, 0, 9, 0, 8]), -np.array([0, 0, 9, 0, 8])),
        ]),
        ("structExploit", [
            (np.array([0, 0, 0, 0, 0]), -np.array([0, 0, 0, 0, 0])),
            (np.array([10, 0, 10, 0, 0]), -np.array([10, 0, 0, 0, 0])),
            (np.array([0, 0, 9, 0, 8]), -np.array([0, 0, 0, 0, 0])),
            (np.array([10, 10, 9, 8, 9]), -np.array([10, 10, 9, 8, 0])),
        ]),
        ("explore", [
            (np.array([0, 0, 0, 0, 0]), -np.array([0, 0, 0, 0, 0])),
            (np.array([10, 0, 10, 0, 0]), -np.array([10, 0, 0, 0, 0])),
            (np.array([0, 0, 9, 0, 8]), -np.array([0, 0, 9, 0, 0])),
            (np.array([10, 10, 9, 8, 9]), -np.array([10, 10, 9, 8, 0]))
        ]),
        ("diversity", [
            (np.array([0, 0, 0, 0, 0]), -np.array([0, 0, 0, 0, 0])),
            (np.array([10, 0, 10, 0, 0]), -np.array([10, 10/2.0, 0, 10/2.0, 10/2.0])),
            (np.array([0, 0, 9, 0, 8]), -np.array([9/2.0, 9/2.0, 9, 9/2.0, 1/2.0])),
            (np.array([10, 10, 9, 8, 9]), -np.array([10, 0, 1/2.0, 2/2.0, 1/2.0]))
        ]),
        ("weakDiversity", [
            (np.array([0, 0, 0, 0, 0]), -np.array([0, 0, 0, 0, 0])),
            (np.array([10, 0, 10, 0, 0]), -np.array([10, 0, 0, 0, 0])),
            (np.array([0, 0, 9, 0, 8]), -np.array([0, 0, 9, 0, 0])),
            (np.array([10, 10, 9, 8, 9]), -np.array([10, 0, 0, 0, 0]))
        ]),
        ("antagonistic", [
            (np.array([0, 0, 0, 0, 0]), -np.array([0, 0, 0, 0, 0])),
            (np.array([10, 0, 10, 0, 0]), -np.array([0, -20, 0, -20, -20])),
            (np.array([0, 0, 9, 0, 8]), -np.array([-17, -17, 1, -17, -1])),
            (np.array([10, 10, 9, 8, 9]), -np.array([-26, -26, -28, -30, -28]))
        ])
])

def test_diagnostic_problem(diagnostic, geno_pheno_pairs):
    for geno, pheno in geno_pheno_pairs:
        out = {}
        problem = DiagnosticProblem(diagnostic=diagnostic, n_var=len(geno), n_obj=len(geno), xl=0, xu=10, damp=1)
        problem._evaluate(geno, out=out)
        assert np.all(out["F"] == pheno)
