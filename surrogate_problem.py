import numpy as np
from pymoo.util.misc import stack
from pymoo.core.problem import Problem

class SurrogateProblem(Problem):

    def __init__(self,nvar,lb,ub,obj_surrogates,const_surrogates=[]):
        super().__init__(n_var=nvar,
                         n_obj=len(obj_surrogates),
                         n_constr=len(const_surrogates),
                         xl=lb,
                         xu=ub)
        self.obj_surrogates = obj_surrogates
        self.const_surrogates = const_surrogates

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = np.column_stack([self.obj_surrogates[i].predict(x) for i in range(len(self.obj_surrogates))])
        if self.n_constr != 0:
            out["G"] = np.column_stack([self.const_surrogates[i].predict(x) for i in range(len(self.const_surrogates))])