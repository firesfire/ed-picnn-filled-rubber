import numpy as np
from .model import ModelState

def harmonic_coefficients(strain,stress):
    n=len(strain);th=2*np.pi*(np.arange(n)+1)/n;amp=.5*(np.max(strain)-np.min(strain))
    gp=2*np.dot(stress,np.sin(th))/n/amp;gpp=2*np.dot(stress,np.cos(th))/n/amp;h={}
    for k in (1,3,5):h[k]=float(np.hypot(2*np.dot(stress,np.sin(k*th))/n,2*np.dot(stress,np.cos(k*th))/n))
    return float(gp),float(gpp),h

def simulate_dma(model,frequency_hz,amplitude,points_per_cycle=128,min_cycles=5,max_cycles=24,steady_tolerance=5e-4):
    dt=1/(frequency_hz*points_per_cycle);state=ModelState.identity(len(model.parameters.branches));prev=None;steady=False;last=None
    for cyc in range(max_cycles):
        strain=[];stress=[]
        for j in range(points_per_cycle):
            q0=amplitude*np.sin(2*np.pi*j/points_per_cycle);q1=amplitude*np.sin(2*np.pi*(j+1)/points_per_cycle)
            F0=model.deformation(q0,"shear");F1=model.deformation(q1,"shear");state,_=model.advance(state,F0,F1,dt);strain.append(q1);stress.append(model.measured_stress(F1,state,"shear"))
        strain=np.asarray(strain);stress=np.asarray(stress);last=(strain,stress)
        if prev is not None and cyc+1>=min_cycles and np.linalg.norm(stress-prev)/max(np.linalg.norm(stress),1e-12)<steady_tolerance:steady=True;break
        prev=stress.copy()
    gp,gpp,h=harmonic_coefficients(*last);area=float(abs(np.trapezoid(last[1],last[0])))
    return {"frequency_hz":frequency_hz,"amplitude":amplitude,"storage_MPa":gp,"loss_MPa":gpp,"tan_delta":gpp/gp,"harmonics_MPa":h,"third_to_first":h[3]/max(h[1],1e-14),"loop_area_MPa":area,"cycles":cyc+1,"steady":steady,"strain_cycle":last[0],"stress_cycle_MPa":last[1]}

def linearized_complex_modulus(model,frequency_hz):
    w=2*np.pi*np.asarray(frequency_hz,float);gp=np.ones_like(w)*model.G0;gpp=np.zeros_like(w)
    for b in model.parameters.branches:
        z=w*b.tau_s;gp+=model.G0*b.g*z*z/(1+z*z);gpp+=model.G0*b.g*z/(1+z*z)
    return gp,gpp

def relaxation_transform(time_s,G,frequency_hz):
    """Discrete Stieltjes transform of a sampled relaxation modulus."""
    t=np.asarray(time_s,float);G=np.asarray(G,float);w=2*np.pi*np.asarray(frequency_hz,float)
    d=-np.diff(G);tm=.5*(t[1:]+t[:-1]);gp=[];gpp=[]
    for om in w:
        gp.append(G[0]-np.sum(d*np.cos(om*tm)))
        gpp.append(np.sum(d*np.sin(om*tm)))
    return np.asarray(gp),np.asarray(gpp)
