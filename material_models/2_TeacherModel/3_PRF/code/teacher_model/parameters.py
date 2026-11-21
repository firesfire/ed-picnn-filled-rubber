from __future__ import annotations
from dataclasses import asdict,dataclass
from typing import Iterable
import json
import numpy as np
@dataclass(frozen=True)
class ZhangLiParameters:
    D:float=.2225397906;mu:float=-.1400176947;B:float=27.47252003;N:float=.9873994149;beta:float=.005474975642
    def validate(self):
        if not(self.D>0 and self.mu<=0 and self.B>=0 and self.N>0 and 0<self.beta<1):raise ValueError('invalid Zhang--Li parameters')
@dataclass(frozen=True)
class FlowParameters:
    gamma0_per_s:float=.12;s0_MPa:float=.12;m:float=2.4;c:float=-.45;h0:float=.08;q0_MPa:float=1e-6
    flow_form:str="timescale_preserving";nonlinear_multiplier:float=0.
    tau_transition_s:float=1.;tau_gate_exponent:float=4.
    def validate(self):
        if self.gamma0_per_s<0 or self.s0_MPa<=0 or self.m<=0 or self.c>0 or self.h0<=0 or self.q0_MPa<=0 or self.nonlinear_multiplier<0 or self.tau_transition_s<=0 or self.tau_gate_exponent<=0:raise ValueError('invalid flow parameters')
        if self.flow_form not in ("timescale_preserving","timescale_hill","bounded_additive","legacy_additive"):raise ValueError('invalid flow form')
@dataclass(frozen=True)
class BranchParameters:
    tau_s:float;g:float
    def validate(self):
        if self.tau_s<=0 or self.g<0:raise ValueError('invalid branch')
@dataclass(frozen=True)
class ModelParameters:
    zhang_li:ZhangLiParameters;flow:FlowParameters;branches:tuple[BranchParameters,...]
    def validate(self):
        self.zhang_li.validate();self.flow.validate();[b.validate() for b in self.branches]
        if np.any(np.diff([b.tau_s for b in self.branches])<=0):raise ValueError('branch times must increase')
    def to_dict(self):return asdict(self)
    def save(self,path):
        with open(path,'w',encoding='utf-8') as f:json.dump(self.to_dict(),f,ensure_ascii=False,indent=2)
    @classmethod
    def from_dict(cls,d):return cls(ZhangLiParameters(**d['zhang_li']),FlowParameters(**d['flow']),tuple(BranchParameters(tau_s=x['tau_s'],g=x['g']) for x in d['branches']))
def build_model_parameters(zl:dict,flow:dict,tau:Iterable[float],weights:Iterable[float],**_):
    p=ModelParameters(ZhangLiParameters(**zl),FlowParameters(**flow),tuple(BranchParameters(float(t),float(g)) for t,g in zip(tau,weights)));p.validate();return p
