#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:15:06 2022
@author: urai
"""

import pandas as pd
import numpy as np

import ssm
from ssm.lds import LDS
from ssm.observations import AutoRegressiveObservations
from ssm.emissions import BernoulliEmissions
from ssm.transitions import StationaryTransitions
from ssm.init_state_distns import InitialStateDistribution

#%% LOAD SOME RANDOM DATA
# from https://figshare.com/articles/dataset/Choice_history_biases_subsequent_evidence_accumulation/7268558/2?file=13593227

data = pd.read_csv('https://figshare.com/ndownloader/files/13593227')
print(data.columns) # we'll use prevpupil as a stand-in for prevconfidence

# ToDo: reorder so it can be used as inputs to ssm

#%% fit with SSM package
# copied from https://codeocean.com/capsule/0617593/tree/v1

class AutoRegressiveNoInput(AutoRegressiveObservations):
    
    def m_step(self, expectations, datas, inputs, masks, tags,
                   continuous_expectations=None, **kwargs):
        
        K, D, M, lags = self.K, self.D, self.M, self.lags

        # Collect sufficient statistics
        if continuous_expectations is None:
            ExuxuTs, ExuyTs, EyyTs, Ens = self._get_sufficient_statistics(expectations, 
                                                                          datas, 
                                                                          inputs)
        else:
            ExuxuTs, ExuyTs, EyyTs, Ens = \
                self._extend_given_sufficient_statistics(expectations, 
                                                         continuous_expectations, 
                                                         inputs)

        Sigmas = np.zeros((K, D, D))
        for k in range(K):
            Wk = np.linalg.solve(ExuxuTs[k] + self.J0[k], ExuyTs[k] + self.h0[k]).T

            # Solve for the MAP estimate of the covariance
            EWxyT =  Wk @ ExuyTs[k]
            sqerr = EyyTs[k] - EWxyT.T - EWxyT + Wk @ ExuxuTs[k] @ Wk.T
            nu = self.nu0 + Ens[k]
            Sigmas[k] = (sqerr + self.Psi0) / (nu + D + 1)

        # If any states are unused, set their parameters to a perturbation of a used state
        unused = np.where(Ens < 1)[0]
        used = np.where(Ens > 1)[0]
        if len(unused) > 0:
            for k in unused:
                i = npr.choice(used)
                Sigmas[k] = Sigmas[i]

        # Update parameters via their setter
        self.Sigmas = Sigmas
        

# ToDo: merge below?
class LDS_noInputDynamics(LDS):
    def __init__(self, N, D, M=0):

        dynamics = AutoRegressiveNoInput(1, D, M=M)
        emissions = BernoulliEmissions(N, 1, D, M=M)

        init_state_distn = InitialStateDistribution(1, D, M)
        transitions = StationaryTransitions(1, D, M)
        
        self.N, self.K, self.D, self.M = N, 1, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        self.emissions = emissions


#%% run the fitting func
stateDim = 1
inputDim = 4
lds = LDS_noInputDynamics(1, 1, M = inputDim)

lds.dynamics.A = np.ones((stateDim,stateDim))   # dynamics
lds.dynamics.b = np.zeros(stateDim)              # bias
lds.dynamics.mu_init = np.zeros((stateDim,stateDim))    # initial mu
lds.dynamics.Sigmas_init = np.array([[[0.01]]])     # initial sigma
lds.dynamics.Vs = np.array([[np.zeros(inputDim)]])               # input dynamics

elbos, q = lds.fit(emissions, inputs = inputs, method="laplace_em",
                    variational_posterior="structured_meanfield", 
                    continuous_optimizer='newton',
                    initialize=True, 
                    num_init_restarts=1,
                    num_iters=200, 
                    alpha=0.1)

# Get the posterior mean of the continuous states (drift)
state_means = q.mean_continuous_states[0]
estDrift = np.squeeze(lds.emissions.Cs)*state_means[:]

# Smooth the data under the variational posterior (to compute emissions)
predEmissions = np.concatenate(lds.smooth(state_means, emissions, input = inputs))


