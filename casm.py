import numpy as np
import GPy
from scipy.optimize import minimize_scalar
from sklearn.utils import resample


def computeBiasedSurrogate(x_train, y_train, sample_y, betal, betau, tol, tau, nboot, niter, fitFunc, evalFunc, mode=True):
    """Summary or Description of the Function

    Parameters:
    x_train (np.array): training points in the active subspace
    y_train (np.array): corresponding values of the QoI
    sample_y (np.array): Description of arg1
    betal (float): lower bound for bias
    betau (float): upper bound for bias
    tol (float): Tolerance for the bisection algorithm
    tau (float): Probability threshold, between 0 and 1
    nboot (int): number of bootstrap replications
    niter (int): max number of bisection iterations
    fitFunc (callable): function to fit a surrogate, with signature (x,y,bias,iteration) -> surrogate
    evalFunc (callable): function to evaluate a surrogate, with signature (f,x) -> evaluations
    mode (bool): Using bootstrap (True, default) or Chernoff bound (False)


    Returns:
    biased_gp: Biased GP surrogate
    beta_previous (float): Bias of the surrogate
   """

    beta_eps = (betau+betal)/2
    err = np.inf
    n = 0
    M_test = np.size(x_train.flatten())

    while err > tol and n<=niter:

        # # Fit biased GPR
        # sig = 0.1
        # gp_g_b = GPy.models.GPRegression(x_train, y_train.reshape(-1, 1)+beta_eps, noise_var=sig)

        # gp_g_b.constrain_positive('')
        # gp_g_b.optimize_restarts(5, verbose=False)
        # print(f"Iteration {n} : fit done")
        gp_g_b = fitFunc(x_train,y_train,beta_eps,n)

        # Generate S sample from training data
        #fhat = gp_g_b.predict(x_train.reshape((M_test,1)),include_likelihood=True)[0].reshape((M_test,1))
        ninactive = sample_y.shape[0]
        fhat = evalFunc(gp_g_b,x_train)
        randix = np.random.randint(0,ninactive,size=M_test)
        # print(randix)
        # print(sample_y[randix,:].diagonal().flatten() - y_train.flatten())
        Sk = fhat.flatten() - sample_y[randix,:].diagonal().flatten()
        avgSk = np.mean(Sk)

        avg = np.zeros(nboot)
        var = np.zeros(nboot)
        if mode: # bootstraping only
            for i in range(nboot):
                avg[i] = np.mean(resample(np.array(Sk.flatten()),n_samples=M_test)>0)
            prob = np.mean(avg)
        else: # chernoff inequality
            # Compute bootstrap estimator for expected value
            for i in range(nboot):
                avg[i] = np.mean(resample(np.array(Sk.flatten()),n_samples=M_test))
                var[i] = np.var(resample(np.array(Sk.flatten()),n_samples=M_test), ddof=1)
            avgSkboot = np.mean(avg)
            varSkboot = np.mean(var)
            # print(avgSkboot)
            # print(varSkboot)
            print(f"Iteration {n} : bootstrap estimator done (Eboot[S] = {avgSkboot}, Vboot[S] = {varSkboot})")

            if(avgSkboot < 0):
                betal = beta_eps
                beta_eps = (betau+betal)/2
                print(f"Iteration {n} : expected value = {avgSkboot} < 0. Increasing bias.")
            else:
                # Approximate Chernoff bound
                def Chernoff():
                    f = lambda x : np.mean(np.exp(x * (np.abs(Sk - avgSkboot) - avgSkboot)))
                    #print((np.abs(Sk - avgSkboot) - avgSkboot))
                    # ax = np.linspace(0,100,100)
                    # plt.plot(ax,list(map(f,ax)))
                    # plt.show()
                    result = minimize_scalar(f,bounds=(0,100),method="bounded")
                    #print(result)
                    return result["fun"]
                
                Ci = np.zeros(1)
                for i in range(1):
                    Ci[i] = Chernoff()
                prob = np.mean(1-Ci)
                print(f"Iteration {n} : concentration bound done")

        err = np.abs(prob-tau)

        beta_previous = beta_eps
        if prob > tau:
            betau = beta_eps
        else:
            betal = beta_eps
        beta_eps = (betau+betal)/2

        print(f"Iteration {n} : bias = {beta_previous}, success probability = {prob}")
            
        n = n + 1


    if (n>25):
        print(f"Iteration {n} : diverged")
        if beta_previous/np.mean(sample_y[randix,:].diagonal().flatten()) < 1e-3 :
            print(f"Bias is small relative to function average. Target probability might be lower than the no-bias conservativeness.")

    return gp_g_b, beta_previous


def computeBiasedGPSurrogate(x_train, y_train, sample_y, betal, betau, tol, tau, nboot, niter, mode=True):

    def fitting(x,y,bias,it):
        # Fit biased GPR
        sig = 0.1
        gp_g_b = GPy.models.GPRegression(x, y.reshape(-1, 1)+bias, noise_var=sig)

        gp_g_b.constrain_positive('')
        gp_g_b.optimize_restarts(5, verbose=False)
        print(f"Iteration {it} : fit done")
        return gp_g_b
    
    def evaluate(f,x):
        M = np.shape(x_train.flatten())[0]
        return f.predict(x_train.reshape((M,1)),include_likelihood=True)[0].reshape((M,1))

    
    return computeBiasedSurrogate(x_train,y_train,sample_y,betal,betau,tol,tau,nboot,niter,fitting,evaluate,mode)




def computeBiasedPolySurrogate(x_train, y_train, sample_y, deg, betal, betau, tol, tau, nboot, niter, mode=True):
    
    def fitting(x,y,bias,it):
        # Fitting biased polynomial
        coeffb = np.polyfit(x.flatten(),y+bias,deg)
        pb = lambda z : np.polyval(coeffb,z) 
        print(f"Iteration {it} : fit done")
        return pb
    
    def evaluate(f,x):
        return f(x.flatten())
    
    return computeBiasedSurrogate(x_train,y_train,sample_y,betal,betau,tol,tau,nboot,niter,fitting,evaluate,mode)