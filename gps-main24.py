    # gps-main9.py

"""
Created on Mon Dec 6, 2021  This version Oct 30, 2022

@author: Alec Kercheval

"""

'''

added: using simulationGPS3

added: loop through parameters and output some descriptive statistics
to a text file

added: outcomes as a function of the angle between beta and tau

added: pathwise differences in angle for boxplots

program to run simulations demonstrating GPS improvements for various metrics

The SimulationGPS class contains the random variable calls that produce beta and
the returns matrix.

Here we have:

ComputePCA_GPS() in which the eigenvalues and eigenvectors are extracted from a supplied
    sample covariance matrix S.
ComputeMRPortfolio() implements the formula for the minimum risk portfolio depending on
input parameters beta, delta, sigma

The main loop iterates through NumExperiments to compute true and
estimated min risk portfolios, tracking error, and
variance forecast ratios.
Choices for comparison are:
raw and corrected evalues against PCA and JSE evectors

This gives 4 combinations to provide 4 min var portfolios for comparison.
We look at tracking error (comparing to true portfolio), VFR (estimated to true risk of estimated portfolio), and true variance ratios (ratio of true variance of true portfolio to true variance of estimated portfolio.)

Tr = (w_est - w)Sigma(w_est - w)
VFR = w_est Sigma_est w_est / w_est Sigma w_est
TVR = w Sigma w / w_est Sigma w_est

Output is boxplot figures summarizing NumExperiments saved to files in the current directory.

Variable notation and formulas come from GPS and MAPS papers.

'''

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#from numpy import random
import simulationGPS3 as sgps
from numpy import linalg as la
from scipy.stats import iqr
import csv



def ComputePCA_GPS(S, Srank, Sdim):

# function to compute PCA and GPS estimators of b given sample cov. matrix S
# and rank Srank, which will equal the number of periods. 
# Sdim = size of S = num of assets.
# There are Srank nonzero evalues, Srank -1 of them below the leading evalue

    evalues, evectors = la.eigh(S)  # S a sym matrix of size MaxAssets, rank Srank
    h = evectors[:, Sdim - 1]  # normalized evector corr. to largest evalue of S
    sp2 = evalues[Sdim - 1]  # leading evalue of S
    lp2 = (np.sum(evalues)-sp2)/(Srank -1) # average of the lesser nonzero evalues
    psi2 = (sp2 - lp2)/sp2  # this is the psi^2 term from the GPS paper
    all_ones = np.ones(Sdim)
    q = all_ones/la.norm(all_ones)  # north pole, unit vector
    
    hq = np.dot(h,q)  # inner product of h and q
    if hq < 0:
        h = -h      # choose e-vector h with positive mean
        hq = -hq
    elif hq == 0:
        print("error: h is orthogonal to q")

    tau = (1-psi2)*hq/(psi2 - hq*hq) # gps data driven shrinkage parameter
    h_shr = h + tau*q   # h_GPS before normalizing
    return h, (1/la.norm(h_shr))*h_shr, sp2, lp2  # h and h_GPS, normalized, and sp2, lp2

###  end def 


def ComputeMRPortfolio(p, p_eta, delta2, h):

# outputs w = argmin w^T Sigma w, subj to w^T e = 1.
# Here Sigma is the real or estimated covariance matrix, depending on inputs
# p = dimension of Sigma = number of assets
# p_eta, delta2, h determine Sigma = p_eta hh^T + delta2 I
# Notation follows MAPS, section 3

    all_ones = np.ones(p)
    q = all_ones/la.norm(all_ones)  # north pole, unit vector
    hq = np.dot(h,q)

    k2 = delta2/p_eta
    rho = (1+k2)/hq
    w = ((rho*q)-h)/((rho - hq)*np.sqrt(p))
    return w

### end def 
    
#####################################################################
### main program   ##################################################
#####################################################################

NumExperiments = 1000
NumPeriods = 40   #  default 252 -- must be less than MaxAssets
Beta_NSqP = 1 # |beta|^2/p  default was 1.25


beta_angle_actual = np.zeros((4,3))
beta_norm2p_actual = np.zeros((4,3))

asset_size_vector = [50,100,200,500]
beta_angle_vector = [0.174, 0.785, 1.396]  # angles in radians

table_rows = np.zeros((4,3,14))

for num_assets in range(4):
    for beta_angle in range(3):

        

    # set up parameters for input to SimulationGPS object

        MaxAssets = asset_size_vector[num_assets] # default 500

        DayString = "d221030p="+str(MaxAssets)


        Beta_Angle_radians = beta_angle_vector[beta_angle]  

        BetaMean = np.around(Beta_NSqP*np.cos(Beta_Angle_radians),decimals=4)
        BetaStDev = np.around(Beta_NSqP*np.sin(Beta_Angle_radians),decimals=4)


        Beta_angle = np.around(Beta_Angle_radians, decimals=3)
        #Beta_normsq_overp = np.around(BetaMean*BetaMean + BetaStDev*BetaStDev, decimals=3)
        Beta_normsq_overp = np.around(Beta_NSqP, decimals=3)

        Factor1StDev = 0.16/np.sqrt(252)
        SpecificStDev = 0.6/np.sqrt(252)  # daily vol from annual

        Factor2StDev = .04/np.sqrt(252) # optional extra factors
        Factor3StDev = .04/np.sqrt(252)
        Factor4StDev = .08/np.sqrt(252)

        FactorFlag = 0  # 0 for one factor; 1 for four factors
        NormalFlag = 0  # 0 for Normal specific returns, 1 for double exponential, 2 for student's t

    
        # create simulationGPS object, which
        # creates beta, factor, specific, total returns  x numExperiments

        rng = np.random.default_rng()  # makes a random number generator, random seed

        sim = sgps.SimulationGPS3(rng, NormalFlag, MaxAssets, NumExperiments, NumPeriods, BetaMean, BetaStDev, Factor1StDev, SpecificStDev, FactorFlag, Factor2StDev, Factor3StDev, Factor4StDev)

        # get returns matrix (assets x periods x numExperiments) from sim object
#        Rtot = sim.GetReturnsMatrix()

        # get true betas -- one beta for all experiments
        betaVector = sim.GetBetaVector()  


        beta_mean_actual = np.mean(betaVector)
        beta_std_actual = np.std(betaVector)

# now scale beta <-- a beta + c  such that angle and norm2/p as as specified
# Given theta and |beta|^2/p = 1, our scale constants are
# a = sin(theta)/s   and  c = cos(theta) - (sin(theta)/s) m
# where m and s are the initial mean and std of beta

        all_ones = np.ones(MaxAssets)
        theta = beta_angle_vector[beta_angle]  # the current target angle (radians)

        a_const = np.sin(theta)/beta_std_actual
        c_const = np.cos(theta) - (np.sin(theta)/beta_std_actual)*beta_mean_actual

        betaVector = a_const*betaVector + c_const*all_ones

        betaVector_unit = betaVector/la.norm(betaVector) 

        q = all_ones/la.norm(all_ones)

        beta_angle_actual[num_assets, beta_angle] = np.arccos(np.dot(betaVector_unit,q))
        beta_norm2p_actual[num_assets, beta_angle] = np.dot(betaVector,betaVector)/MaxAssets

        Rtot = sim.CreateReturnsMatrix(betaVector)


        # set up arrays to hold results of metrics per trial
        # subscript key: T_ = true evalues, E_ = estimated evalues  _T,_pca,_jse refer to evectors

        VFR_raw = np.zeros((NumExperiments))   # var forecast ratio 
        #VFR_jse = np.zeros((NumExperiments))   # var forecast ratio 
        VFR_Epca = np.zeros((NumExperiments))   # var forecast ratio 
        VFR_Ejse = np.zeros((NumExperiments))   # var forecast ratio 

        TrueVarR_Epca = np.zeros((NumExperiments))  # ratio of true min var to true var of est portfolio
        TrueVarR_Ejse = np.zeros((NumExperiments))
        TrueVarR_raw = np.zeros((NumExperiments))
        #TrueVarR_jse = np.zeros((NumExperiments))

        trerrorraw = np.zeros((NumExperiments)) # pca tracking error
        #trerrorjse = np.zeros((NumExperiments)) # pca tracking error
        trerrorEpca = np.zeros((NumExperiments)) # pca tracking error
        trerrorEjse = np.zeros((NumExperiments))  # gps tracking error

        trueVar_Epca = np.zeros((NumExperiments)) # true variance of the PCA portfolio (min var with PCA cov estimate)
        trueVar_Ejse = np.zeros((NumExperiments))    # true variance of the GPS portfolio
        trueVar_raw = np.zeros((NumExperiments))
        #trueVar_jse = np.zeros((NumExperiments))
        trueVar_TT = np.zeros((NumExperiments))

        estVar_Epca = np.zeros((NumExperiments)) # estimated variance of the PCA portfolio (min var with PCA cov estimate)
        estVar_Ejse = np.zeros((NumExperiments))    # est variance of the GPS portfolio
        estVar_raw = np.zeros((NumExperiments))
        #estVar_jse = np.zeros((NumExperiments))

        angle_PCA = np.zeros((NumExperiments)) # angle error for PCA estimate of b
        angle_JSE = np.zeros((NumExperiments)) # angle error for JSE estimate of b
        angle_diff = np.zeros((NumExperiments)) # pathwise difference between angle errors




        # main loop iterating trials

        for exper in range(NumExperiments):  
            Y = Rtot[:,:,exper]  # matrix of returns for trial exper
            S = np.matmul(Y,Y.transpose())/NumPeriods  # sample covariance matrix for trial exper
            b = betaVector/la.norm(betaVector)  # normalized beta

            h, h_GPS, sp2, lp2 = ComputePCA_GPS(S, NumPeriods, MaxAssets) # defined above



            angle_PCA[exper] = np.arccos(np.dot(h,b))
            angle_JSE[exper] = np.arccos(np.dot(h_GPS, b))
            angle_diff[exper] = angle_PCA[exper] - angle_JSE[exper]
        #    angle_bq[exper] = np.arccos(np.dot(b,q))

            # tracking error notation and formulas from MAPS paper, section 3

            p_eta_true = np.dot(betaVector,betaVector)*(Factor1StDev)**2
            p_eta_obs = sp2 - lp2
            delta2_true = (SpecificStDev)**2
            delta2_obs = (NumPeriods/MaxAssets)*lp2


            delta2_raw = ((NumPeriods-1)/(MaxAssets-1))*lp2
            p_eta_raw = sp2 - delta2_raw
          # see the JS paper for the raw PCA estimator

            #  four portfolios	

            w_Epca = ComputeMRPortfolio(MaxAssets, p_eta_obs, delta2_obs, h)    # estimated evalues and PCA evector
            w_Ejse = ComputeMRPortfolio(MaxAssets, p_eta_obs, delta2_obs, h_GPS) # estimated evalues and GPS evector

            w_raw = ComputeMRPortfolio(MaxAssets, p_eta_raw, delta2_raw, h)

            w_pca = ComputeMRPortfolio(MaxAssets, p_eta_obs, delta2_raw, h) # true evalues, PCA evector
            w_jse = ComputeMRPortfolio(MaxAssets, p_eta_obs, delta2_raw, h_GPS) # true evalues, GPS evector

            w_TT = ComputeMRPortfolio(MaxAssets, p_eta_true, delta2_true, b)



            TrackErr2_Epca = p_eta_true*((np.dot(w_Epca - w_TT, b))**2) + delta2_true*((la.norm(w_Epca - w_TT))**2)

            TrackErr2_Ejse = p_eta_true*((np.dot(w_Ejse - w_TT, b))**2) + delta2_true*((la.norm(w_Ejse - w_TT))**2)

 
            TrackErr2_raw = p_eta_true*((np.dot(w_raw - w_TT, b))**2) + delta2_true*((la.norm(w_raw - w_TT))**2)
 
        #    TrackErr2_jse = p_eta_true*((np.dot(w_jse - w_TT, b))**2) + delta2_true*((la.norm(w_jse - w_TT))**2)

            trerrorEpca[exper] = np.sqrt(252*TrackErr2_Epca)*100
            trerrorEjse[exper] = np.sqrt(252*TrackErr2_Ejse)*100
            trerrorraw[exper] = np.sqrt(252*TrackErr2_raw)*100
        #    trerrorjse[exper] = np.sqrt(252*TrackErr2_jse)*100

            # variance forecast ratios = estimated / true variances of the estimated portfolio
            # see GPS and MAPS for these variance formulas

            trueVar_Epca[exper] = p_eta_true*((np.dot(b,w_Epca))**2) + delta2_true*np.dot(w_Epca,w_Epca)
            trueVar_Ejse[exper] = p_eta_true*((np.dot(b,w_Ejse))**2) + delta2_true*np.dot(w_Ejse,w_Ejse)    
            trueVar_raw[exper] = p_eta_true*((np.dot(b,w_raw))**2) + delta2_true*np.dot(w_raw,w_raw)
        #    trueVar_jse[exper] = p_eta_true*((np.dot(b,w_jse))**2) + delta2_true*np.dot(w_jse,w_jse)
            trueVar_TT[exper] = p_eta_true*((np.dot(b,w_TT))**2) + delta2_true*np.dot(w_TT,w_TT)

            estVar_Epca[exper] = p_eta_obs*((np.dot(h,w_Epca))**2) + delta2_obs*np.dot(w_Epca,w_Epca)
            estVar_Ejse[exper] = p_eta_obs*((np.dot(h_GPS,w_Ejse))**2) + delta2_obs*np.dot(w_Ejse,w_Ejse)
            estVar_raw[exper] = p_eta_raw*((np.dot(h,w_raw))**2) + delta2_raw*np.dot(w_raw,w_raw)



            VFR_Epca[exper] = estVar_Epca[exper]/trueVar_Epca[exper]
            VFR_Ejse[exper] = estVar_Ejse[exper]/trueVar_Ejse[exper]
            VFR_raw[exper] = estVar_raw[exper]/trueVar_raw[exper]


            TrueVarR_Epca[exper] =  trueVar_TT[exper]/trueVar_Epca[exper]
            TrueVarR_Ejse[exper] =  trueVar_TT[exper]/trueVar_Ejse[exper]
            TrueVarR_raw[exper] =  trueVar_TT[exper]/trueVar_raw[exper]


        
        #endfor exper

        angle_diff_mean = np.around(np.mean(angle_diff),decimals=3)
        angle_diff_median = np.around(np.median(angle_diff),decimals=3)

        angle_mean_pca = np.mean(angle_PCA)
        angle_median_pca = np.median(angle_PCA)
        angle_iqr_pca = iqr(angle_PCA)

        angle_mean_jse = np.mean(angle_JSE)
        angle_median_jse = np.median(angle_JSE)
        angle_iqr_jse = iqr(angle_JSE)

        count = 0
        for exper in range(NumExperiments):
            if angle_diff[exper] >= 0:
                count += 1

        prob_diff_pos = np.around(count/NumExperiments, decimals=3)

        table_rows[num_assets, beta_angle,:] = [asset_size_vector[num_assets], beta_angle_vector[beta_angle], beta_angle_actual[num_assets, beta_angle], beta_norm2p_actual[num_assets, beta_angle], angle_diff_mean, angle_diff_median, iqr(angle_diff), prob_diff_pos, angle_mean_pca, angle_median_pca, angle_iqr_pca, angle_mean_jse, angle_median_jse, angle_iqr_jse]

        ### output data files
        
        # np.savetxt(DayString+'varFR_PCA.out',varFR_PCA, fmt='%.3e', delimiter=',')
        # np.savetxt(DayString+'varFR_GPS.out', varFR_GPS, fmt='%.3e', delimiter=',')
        # np.savetxt(DayString+'anglePCA.out',anglePCA, fmt='%.3e', delimiter=',')
        # np.savetxt(DayString+'angleGPS.out', angleGPS, fmt='%.3e', delimiter=',')
        # np.savetxt(DayString+'trerrorPCA.out',trerrorPCA, fmt='%.3e', delimiter=',')
        # np.savetxt(DayString+'trerrorGPS.out', trerrorGPS,fmt='%.3e', delimiter=',')
    
        # np.savetxt(DayString+'beta_angle'+str(Beta_Angle_degrees)+'angle_PCA.out', angle_PCA, fmt='%.3e', delimiter=',')
        # np.savetxt(DayString+'beta_angle'+str(Beta_Angle_degrees)+'angle_JSE.out', angle_JSE, fmt='%.3e', delimiter=',')
        # np.savetxt(DayString+'beta_angle'+str(Beta_Angle_degrees)+'angle_diff.out', angle_diff, fmt='%.3e', delimiter=',')

        ### output box plots  ######################################

        # column_names = [str(MaxAssets)+'\nEpca', str(MaxAssets)+'\nEjse', str(MaxAssets)+'\nET', str(MaxAssets)+'\nTpca', str(MaxAssets)+'\nTjse', str(MaxAssets)+'\nTT']
        column_names = ['raw', 'PCA', 'JSE']
        angle_column_names = ['PCA', 'JSE']



        #### angle error differences box plot

        data_angle_diff = np.array([angle_diff]).transpose()
        #df_angle = pd.DataFrame(data_angle, columns=angle_column_names)
        df_angle_diff = pd.DataFrame(data_angle_diff)

        FigAngleDiff = plt.figure()
        bp_angle_diff = df_angle_diff.boxplot()
        plt.ylabel("pathwise angle error differences (radions)")
        #plt.xlabel("estimator")
        plt.title(str(MaxAssets)+" assets, diff mean = "+str(angle_diff_mean)+", diff median = "+str(angle_diff_median)+", P(pos) = "+str(prob_diff_pos)+", beta angle = "+str(Beta_angle)+", |beta|^2/p = "+str(Beta_normsq_overp))

        FigAngleDiff.savefig("output/"+DayString+"angle_diff_boxplot_E"+str(NumExperiments)+"T"+str(NumPeriods)+"f"+str(NormalFlag)+"bm"+str(BetaMean)+"bs"+str(BetaStDev)+"A"+str(MaxAssets)+".png", format="png", bbox_inches="tight")

        #### angle error box plot

        data_angle = np.array([angle_PCA, angle_JSE]).transpose()
        df_angle = pd.DataFrame(data_angle, columns=angle_column_names)

        FigAngle = plt.figure()
        bp_angle = df_angle.boxplot()
        plt.ylabel("angle error (radions)")
        plt.xlabel("estimator")
        plt.title(str(MaxAssets)+" assets, beta mean = "+str(BetaMean)+", beta Std Dev = "+str(BetaStDev)+", beta angle = "+str(Beta_angle)+", |beta|^2/p = "+str(Beta_normsq_overp))

        FigAngle.savefig("output/"+DayString+"angle_boxplot_E"+str(NumExperiments)+"T"+str(NumPeriods)+"f"+str(NormalFlag)+"bm"+str(BetaMean)+"bs"+str(BetaStDev)+"A"+str(MaxAssets)+".png", format="png", bbox_inches="tight")


        #### tracking error

        #data_tr = np.array([trerrorET, trerrorEpca, trerrorTpca, trerrorEjse, trerrorTjse]).transpose()
        data_tr = np.array([trerrorraw, trerrorEpca, trerrorEjse]).transpose()

        df_tr = pd.DataFrame(data_tr, columns=column_names)

        FigTrE = plt.figure()
        bp_tr = df_tr.boxplot()
        plt.ylabel("annualized tracking error (%)")
        plt.xlabel("estimator")
        plt.title(str(MaxAssets)+" assets, beta mean = "+str(BetaMean)+", beta Std Dev = "+str(BetaStDev)+", beta angle = "+str(Beta_angle)+", |beta|^2/p = "+str(Beta_normsq_overp))
        #plt.title("tracking error for E, T = "+str(NumExperiments)+", "+str(NumPeriods))[1]
        FigTrE.savefig("output/"+DayString+"trerrorboxplot_E"+str(NumExperiments)+"T"+str(NumPeriods)+"f"+str(NormalFlag)+"bm"+str(BetaMean)+"bs"+str(BetaStDev)+"A"+str(MaxAssets)+".png", format="png", bbox_inches="tight")

        #### variance forecast ratio

        data_vf = np.array([VFR_raw, VFR_Epca, VFR_Ejse]).transpose()

        df_vf = pd.DataFrame(data_vf, columns=column_names)

        FigVF = plt.figure()
        bp_vf = df_vf.boxplot()
        plt.ylabel("variance forecast ratio")
        plt.xlabel("estimator")
        plt.title(str(MaxAssets)+" assets, beta mean = "+str(BetaMean)+", beta Std Dev = "+str(BetaStDev)+", beta angle = "+str(Beta_angle)+", |beta|^2/p = "+str(Beta_normsq_overp))
        #plt.title("variance forecast ratio for E, T = "+str(NumExperiments)+", "+str(NumPeriods))
        FigVF.savefig("output/"+DayString+"varFR_boxplot_E"+str(NumExperiments)+"T"+str(NumPeriods)+"f"+str(NormalFlag)+"bm"+str(BetaMean)+"bs"+str(BetaStDev)+"A"+str(MaxAssets)+".png", format="png", bbox_inches="tight")


        #### true variance ratios

        data_var = np.array([TrueVarR_raw, TrueVarR_Epca,  TrueVarR_Ejse]).transpose()

        df_var = pd.DataFrame(data_var, columns=column_names)

        FigVar = plt.figure()
        bp_var = df_var.boxplot()
        plt.ylabel("true variance ratio")
        plt.xlabel("estimator")
        plt.title(str(MaxAssets)+" assets, beta mean = "+str(BetaMean)+", beta Std Dev = "+str(BetaStDev)+", beta angle = "+str(Beta_angle)+", |beta|^2/p = "+str(Beta_normsq_overp))

        FigVar.savefig("output/"+DayString+"trueVarRatio_boxplot_E"+str(NumExperiments)+"T"+str(NumPeriods)+"f"+str(NormalFlag)+"bm"+str(BetaMean)+"bs"+str(BetaStDev)+"A"+str(MaxAssets)+".png", format="png", bbox_inches="tight")

with open('results2.csv', 'w', newline='') as csvfile: 
    mywriter = csv.writer(csvfile, dialect='excel')
    mywriter.writerow(['p', 'angle_beta', 'angle_beta_actual', '|beta|^2/p actual', 'mean_diff', 'median_diff', 'iqr_diff', 'P(diff>0)', 'mean_pca', 'median_pca', 'iqr_pca', 'mean_jse', 'median_jse', 'iqr_jse'])
    mywriter.writerow(table_rows[0,0,:])
    mywriter.writerow(table_rows[0,1,:])
    mywriter.writerow(table_rows[0,2,:])
    mywriter.writerow(table_rows[1,0,:])
    mywriter.writerow(table_rows[1,1,:])
    mywriter.writerow(table_rows[1,2,:])
    mywriter.writerow(table_rows[2,0,:])
    mywriter.writerow(table_rows[2,1,:])
    mywriter.writerow(table_rows[2,2,:])
    mywriter.writerow(table_rows[3,0,:])
    mywriter.writerow(table_rows[3,1,:])
    mywriter.writerow(table_rows[3,2,:])


