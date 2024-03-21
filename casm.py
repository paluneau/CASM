import numpy as np
import GPy
from scipy.optimize import minimize_scalar
from sklearn.utils import resample



def computeBiasedSurrogate(x_train, y_train, sample_y, betal, betau, tol, tau, mode=True):
    """Summary or Description of the Function

    Parameters:
    x_train (np.array): training points in the active subspace
    y_train (np.array): corresponding values of the QoI
    sample_y (np.array): Description of arg1
    betal (float): lower bound for bias
    betau (float): upper bound for bias
    tol (float): Tolerance for the bisection algorithm
    tau (float): Probability threshold, between 0 and 1
    mode (bool): Using bootstrap (True, default) or Chernoff bound (False)

    Returns:
    biased_gp: Biased GP surrogate
    beta_previous (float): Bias of the surrogate
   """

    beta_eps = (betau+betal)/2
    err = np.inf
    n = 0
    M_test = np.size(x_train.flatten())

    while err > tol and n<=25:

        # Fit biased GPR
        sig = 0.1
        gp_g_b = GPy.models.GPRegression(x_train, y_train.reshape(-1, 1)+beta_eps, noise_var=sig)

        gp_g_b.constrain_positive('')
        gp_g_b.optimize_restarts(5, verbose=False)
        print(f"Iteration {n} : fit done")

        # Generate S sample from training data
        fhat = gp_g_b.predict(x_train.reshape((M_test,1)),include_likelihood=True)[0].reshape((M_test,1))
        randix = np.random.randint(0,25,size=M_test)
        Sk = fhat.flatten() - sample_y[randix,:].diagonal().flatten()
        avgSk = np.mean(Sk)

        nboot = 5000
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
