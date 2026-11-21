from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, logm
from .flow import dev

SQ2=np.sqrt(2.);SQ6=np.sqrt(6.)
BASIS=np.array([
 [[1/SQ2,0,0],[0,-1/SQ2,0],[0,0,0]],
 [[1/SQ6,0,0],[0,1/SQ6,0],[0,0,-2/SQ6]],
 [[0,1/SQ2,0],[1/SQ2,0,0],[0,0,0]],
 [[0,0,1/SQ2],[0,0,0],[1/SQ2,0,0]],
 [[0,0,0],[0,0,1/SQ2],[0,1/SQ2,0]]])

def vec_to_symdev(x): return np.tensordot(x,BASIS,axes=1)
def symdev_to_vec(A):
    A=dev(.5*(A+A.T));return np.array([np.tensordot(A,b) for b in BASIS])

@dataclass
class NewtonOptions:
    abs_tol:float=1e-10;rel_tol:float=1e-8;max_iter:int=20;jacobian_step:float=1e-7;max_subdivisions:int=10
@dataclass
class StepResult:
    Fv:np.ndarray;converged:bool;iterations:int;residual_norm:float;subdivisions:int;diagnostics:dict

def implicit_exponential_step(Fv0,F1,dt,rate_function,options):
    if dt<=0:return StepResult(Fv0.copy(),True,0,0,0,{})
    D0,_=rate_function(F1,Fv0);z=dt*symdev_to_vec(D0)
    if np.linalg.norm(z)>.5:z*=.5/np.linalg.norm(z)
    def ev(x):
        Fv=expm(vec_to_symdev(x))@Fv0;Dv,d=rate_function(F1,Fv)
        return x-dt*symdev_to_vec(Dv),Fv,d
    r,Fv,d=ev(z);scale=max(np.linalg.norm(r),1.)
    for it in range(1,options.max_iter+1):
        rn=np.linalg.norm(r)
        if rn<=options.abs_tol+options.rel_tol*scale:return StepResult(Fv,True,it,float(rn),0,d)
        J=np.empty((5,5))
        for j in range(5):
            h=options.jacobian_step*(1+abs(z[j]));q=np.zeros(5);q[j]=h
            J[:,j]=(ev(z+q)[0]-ev(z-q)[0])/(2*h)
        try:s=np.linalg.solve(J,-r)
        except np.linalg.LinAlgError:s=np.linalg.lstsq(J,-r,rcond=None)[0]
        accepted=False;alpha=1.
        for _ in range(10):
            zn=z+alpha*s;rnxt,Fn,dn=ev(zn)
            if np.linalg.norm(rnxt)<(1-1e-4*alpha)*rn:
                z,r,Fv,d=zn,rnxt,Fn,dn;accepted=True;break
            alpha*=.5
        if not accepted:break
    return StepResult(Fv,False,options.max_iter,float(np.linalg.norm(r)),0,d)

def isochoric_midpoint(F0,F1):
    """Multiplicative midpoint preserving the isochoric loading manifold."""
    try:
        relative=F1@np.linalg.inv(F0)
        Fm=expm(.5*logm(relative))@F0
        Fm=np.real_if_close(Fm,tol=1000)
        if np.iscomplexobj(Fm) or not np.all(np.isfinite(Fm)):
            raise ValueError("non-real logarithmic midpoint")
        Fm=np.asarray(Fm,float)
    except (ValueError,np.linalg.LinAlgError):
        Fm=.5*(F0+F1)
    det=float(np.linalg.det(Fm))
    if det<=0 or not np.isfinite(det):
        raise ValueError("nonpositive midpoint determinant")
    target=np.sqrt(float(np.linalg.det(F0))*float(np.linalg.det(F1)))
    return Fm*(target/det)**(1./3.)

def integrate_with_subdivision(Fv0,F0,F1,dt,rate_function,options,level=0):
    r=implicit_exponential_step(Fv0,F1,dt,rate_function,options)
    if r.converged:r.subdivisions=level;return r
    if level>=options.max_subdivisions:raise RuntimeError(f"Newton failed residual={r.residual_norm:.3e}, dt={dt:.3e}")
    Fm=isochoric_midpoint(F0,F1)
    a=integrate_with_subdivision(Fv0,F0,Fm,dt/2,rate_function,options,level+1)
    b=integrate_with_subdivision(a.Fv,Fm,F1,dt/2,rate_function,options,level+1)
    b.iterations+=a.iterations;b.subdivisions=max(a.subdivisions,b.subdivisions);return b
