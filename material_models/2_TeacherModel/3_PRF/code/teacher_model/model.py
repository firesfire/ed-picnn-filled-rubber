from dataclasses import dataclass
import numpy as np
from .zhang_li import energy,mandel,small_strain_moduli,unpressurized_cauchy
from .flow import dev,flow_rate
from .integrator import NewtonOptions,integrate_with_subdivision
@dataclass
class ModelState:
    Fv:list[np.ndarray]
    @classmethod
    def identity(cls,n):return cls([np.eye(3) for _ in range(n)])
    def copy(self):return ModelState([x.copy() for x in self.Fv])
class TeacherModel:
    def __init__(self,parameters,newton=None):parameters.validate();self.parameters=parameters;self.newton=newton or NewtonOptions();self.G0,self.E0=small_strain_moduli(parameters.zhang_li)
    def _branch(self,F,Fv,b):
        Fe=F@np.linalg.inv(Fv);We=energy(Fe.T@Fe,self.parameters.zhang_li);M=mandel(Fe,self.parameters.zhang_li,b.g);X=dev(M);eta=max(3*b.g*self.G0*b.tau_s,1e-14);Dv,d=flow_rate(X,Fv,eta,self.parameters.flow,1.,b.g,self.G0,b.tau_s);return Dv,d,unpressurized_cauchy(Fe,self.parameters.zhang_li,b.g),Fe,X,We
    def advance(self,state,F0,F1,dt):
        new=[];diag=[]
        for Fv,b in zip(state.Fv,self.parameters.branches):
            def rate(F,V):return self._branch(F,V,b)[:2]
            r=integrate_with_subdivision(Fv,F0,F1,dt,rate,self.newton);new.append(r.Fv);diag.append({**r.diagnostics,'newton_iterations':r.iterations,'residual':r.residual_norm,'subdivisions':r.subdivisions,'detFv':float(np.linalg.det(r.Fv))})
        return ModelState(new),diag
    def response(self,F,state):
        sig=unpressurized_cauchy(F,self.parameters.zhang_li);bs=[];bd=[];W=energy(F.T@F,self.parameters.zhang_li)
        for Fv,b in zip(state.Fv,self.parameters.branches):
            Dv,d,s,Fe,X,We=self._branch(F,Fv,b);sig+=s;bs.append(s);bd.append({**d,'X':X,'Dv':Dv});W+=b.g*We
        return {'cauchy_unpressurized':sig,'branch_cauchy':bs,'branch':bd,'energy_MPa':float(W)}
    @staticmethod
    def deformation(value,mode):
        if mode=='uniaxial':
            l=1+value
            if l<=0:raise ValueError('nonpositive stretch')
            return np.diag([l,l**-.5,l**-.5])
        if mode=='shear':return np.array([[1.,value,0.],[0.,1.,0.],[0.,0.,1.]])
        raise ValueError(mode)
    def measured_stress(self,F,state,mode):
        s=self.response(F,state)['cauchy_unpressurized']
        if mode=='uniaxial':return float((s[0,0]-.5*(s[1,1]+s[2,2]))/F[0,0])
        if mode=='shear':return float(s[0,1])
        raise ValueError(mode)
