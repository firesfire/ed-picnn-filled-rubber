import numpy as np
from .model import ModelState

def simulate_history(model,time_s,scalar_history,mode,max_scalar_increment=.025,return_states=False):
    t=np.asarray(time_s,float);x=np.asarray(scalar_history,float)
    if len(t)!=len(x) or np.any(np.diff(t)<=0):raise ValueError("strictly increasing time required")
    state=ModelState.identity(len(model.parameters.branches));F=model.deformation(x[0],mode)
    stress=[model.measured_stress(F,state,mode)];diag=[];states=[state.copy()] if return_states else None
    for k in range(1,len(t)):
        dt=t[k]-t[k-1];n=max(1,int(np.ceil(abs(x[k]-x[k-1])/max_scalar_increment)))
        for j in range(n):
            a=j/n;b=(j+1)/n;F0=model.deformation((1-a)*x[k-1]+a*x[k],mode);F1=model.deformation((1-b)*x[k-1]+b*x[k],mode)
            state,d=model.advance(state,F0,F1,dt/n);diag.extend(d)
        F=model.deformation(x[k],mode);stress.append(model.measured_stress(F,state,mode))
        if return_states:states.append(state.copy())
    out={"time_s":t,"strain":x,"nominal_stress_MPa":np.array(stress),"diagnostics":diag}
    if return_states:out["states"]=states
    return out
def simulate_relaxation_curve(model,curve,**kwargs):return simulate_history(model,curve.time_s,curve.strain,curve.mode,**kwargs)

