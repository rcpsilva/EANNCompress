import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem

class SurrogateProblem(Problem):

    def __init__(self,nvar,obj_surrogate,const_surrogate,lb,ub):
        super().__init__(n_var=nvar,
                         n_obj=len(obj_surrogate),
                         n_constr=len(const_surrogate),
                         xl=lb,
                         xu=ub,
                         elementwise_evaluation=True)
        self.obj_surrogate = obj_surrogate
        self.const_surrogate = const_surrogate

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = self.obj_surrogate(x)
        out["G"] = self.const_surrogate(x)