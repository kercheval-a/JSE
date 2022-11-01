#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 15:13:44 2021
simulationGPS2.py
New version updated October 30, 2022

@author: alec kercheval
"""

'''
class to support simulation experiments for GPS
Creates betas, factor returns, specific returns, and total returns
for each of NumExperiments experimental runs.
All randomness occurs here.  Computations appear in the main program.

In addition to beta, we add the option to consider a four-factor model
with beta, loading2, loading3, loading4 and corresponding factors:
factor, factor2, factor3, factor 4.
Beta and three extra loadings will be generated once and stay constant across experiments.
'''

import numpy as np
#from numpy import random
#from numpy import linalg

class SimulationGPS3:
    'simulation class for GPS estimation '

    def __init__(self, Rng, NormalFlag, MaxAssets, NumExperiments, NumPeriods, BetaMean=1, BetaStDev=0.5, Factor1StDev=0.16/np.sqrt(252), SpecificStDev=0.6/np.sqrt(252),  FactorFlag = 0, Factor2StDev = .04/np.sqrt(252), Factor3StDev = .04/np.sqrt(252), Factor4StDev = .08/np.sqrt(252)):
        self.rng = Rng  # random number generator
        self.maxAssets = MaxAssets # maximum number of assets in the simulation
        self.numExperiments = NumExperiments # number of trials
        self.numPeriods = NumPeriods # length of sample (days)
        self.betaMean = BetaMean # mean of distribution of betas
        self.betaStDev = BetaStDev # st deviation of betas
        
        self.factor1StDev = Factor1StDev # std dev of beta factor (mean zero)
        self.factor2StDev = Factor2StDev
        self.factor3StDev = Factor3StDev
        self.factor4StDev = Factor4StDev
        self.specificStDev = SpecificStDev # std dev of spec return (mean 0)

        self.normalFlag = NormalFlag # 0 for normal specific returns
        self.factorFlag = FactorFlag # 0 for one factor, 1 for four factors
        
        self.betas = self.CreateBetas()
            # populate vector of betas of size maxAssets, fixed across experiments
#        self.exposures234 = self.CreateExposures234()  
            # create maxAssets x 3 matrix of exposures for the other three factors

        self.factor1Matrix = self.CreateFactors1()
            #populate factor matrix of size NumPeriods x numExperiments

#        self.factor234Matrix = self.CreateFactors234()
            #populate factor matrix of size 3 x NumPeriods x numExperiments

        self.specMatrix = self.CreateSpecMatrix()
            # populate array of specific returns of size 
            # numAssets x numPeriods x numExperiments
#        self.returnsMatrix = self.CreateReturnsMatrix()

    # end def __init__



    # create N-dim beta and exposure vectors
    def CreateBetas(self):
        return self.rng.normal(self.betaMean, self.betaStDev, self.maxAssets)

    def CreateExposures234(self):
        # only called if factorFlag is not zero, size maxAssets x 3 
        return self.rng.normal(0, 1, (self.maxAssets, 3))
        

    # create periods x E factor matrix
    def CreateFactors1(self):
        return self.rng.normal(0, self.factor1StDev, (self.numPeriods, self.numExperiments))

    # create 3 x periods x E factor matrix
    def CreateFactors234(self):
        factors234 = np.zeros((3, self.numPeriods, self.numExperiments)) # initialize
        row1 = self.rng.normal(0, self.factor2StDev, (self.numPeriods, self.numExperiments))
        row2 = self.rng.normal(0, self.factor3StDev, (self.numPeriods, self.numExperiments))
        row3 = self.rng.normal(0, self.factor4StDev, (self.numPeriods, self.numExperiments))
        factors234[0,:,:] = row1
        factors234[1,:,:] = row2
        factors234[2,:,:] = row3
        return factors234

    # create assets x periods x E specific returns matrix
    def CreateSpecMatrix(self):
        if self.normalFlag == 0:
            return self.rng.normal(0, self.specificStDev, (self.maxAssets, self.numPeriods, self.numExperiments))
        elif self.normalFlag == 1:
            return self.rng.laplace(0, self.specificStDev/np.sqrt(2), (self.maxAssets, self.numPeriods, self.numExperiments) )
        elif self.normalFlag == 2: 
            df = 5  # degrees of freedom
            multiplier = np.sqrt((df-2)/df)*self.specificStDev # var = df/(df-2)
            return multiplier*self.rng.standard_t(df, (self.maxAssets, self.numPeriods, self.numExperiments) )

    def CreateReturnsMatrix(self, b):  # input beta vector to use
        N = self.maxAssets
        T = self.numPeriods
        Ex = self.numExperiments
#        b = self.betas
        self.betas = b
        F = self.factor1Matrix
        Z = self.specMatrix
        Rtot = np.zeros((N,T,Ex))  # matrix to hold returns for each of Ex experiments
        if self.factorFlag == 0:
            for e in range(Ex):
                f = F[:,e]
                z = Z[:,:,e]

                R = np.outer(b,f) + z  # one factor model of returns
                        # outer product of N dim b and T dim f
                Rtot[:,:,e] = R[:,:]
        else:
            bb = self.CreateExposures234()
            F234 = self.CreateFactors234()
            for e in range(Ex):
                f = F[:,e]
                z = Z[:,:,e]
                ff = F234[:,:,e]

                R = np.outer(b,f) + np.matmul(bb,ff) + z  # four factor model
                Rtot[:,:,e] = R[:,:]
                
        return Rtot

#    def GetReturnsMatrix(self):
#        return self.returnsMatrix

    def GetBetaVector(self):
        return self.betas



