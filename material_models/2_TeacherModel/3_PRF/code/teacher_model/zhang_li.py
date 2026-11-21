import numpy as np

def invariants(C):
    i1=float(np.trace(C));i2=0.5*(i1*i1-float(np.trace(C@C)))
    if not np.isfinite(i1) or not np.isfinite(i2):
        raise FloatingPointError("non-finite elastic invariants")
    return i1,i2
def energy(C,p,normalized=True):
    i1,i2=invariants(C);base=i2-3+p.N
    if base<=0:raise FloatingPointError("non-positive Zhang--Li I2 base in Newton trial state")
    with np.errstate(over="raise",invalid="raise"):
        a=p.D*(i1-3) if abs(p.mu)<1e-12 else p.D/p.mu*(1-np.exp(-p.mu*(i1-3)));b=p.B*base**p.beta
    return float(a+b-(p.B*p.N**p.beta if normalized else 0))
def derivatives(C,p):
    i1,i2=invariants(C);base=i2-3+p.N
    if base<=0:raise FloatingPointError("non-positive Zhang--Li I2 base in Newton trial state")
    with np.errstate(over="raise",invalid="raise"):
        return float(p.D*np.exp(-p.mu*(i1-3))),float(p.B*p.beta*base**(p.beta-1))
def unpressurized_cauchy(Fe,p,weight=1):
    B=Fe@Fe.T;w1,w2=derivatives(Fe.T@Fe,p);return weight*(2*w1*B-2*w2*np.linalg.inv(B))
def mandel(Fe,p,weight=1):
    C=Fe.T@Fe;i1,_=invariants(C);w1,w2=derivatives(C,p);M=2*weight*(w1*C+w2*(i1*C-C@C));return .5*(M+M.T)
def nominal_analytic(stretch,mode,p):
    l=np.asarray(stretch,float)
    if mode=="uniaxial": i1=l*l+2/l;i2=l**-2+2*l;w1=p.D*np.exp(-p.mu*(i1-3));w2=p.B*p.beta*(i2-3+p.N)**(p.beta-1);return 2*(l-l**-2)*(w1+w2/l)
    if mode=="planar": i=l*l+1+l**-2;w1=p.D*np.exp(-p.mu*(i-3));w2=p.B*p.beta*(i-3+p.N)**(p.beta-1);return 2*(l-l**-3)*(w1+w2)
    if mode=="equibiaxial": i1=2*l*l+l**-4;i2=l**4+2*l**-2;w1=p.D*np.exp(-p.mu*(i1-3));w2=p.B*p.beta*(i2-3+p.N)**(p.beta-1);return 2*(l-l**-5)*(w1+l*l*w2)
    raise ValueError(mode)
def simple_shear_analytic(gamma,p):
    g=np.asarray(gamma,float);i=3+g*g;return 2*g*(p.D*np.exp(-p.mu*(i-3))+p.B*p.beta*(i-3+p.N)**(p.beta-1))
def small_strain_moduli(p):
    g=2*(p.D+p.B*p.beta*p.N**(p.beta-1));return float(g),float(3*g)
