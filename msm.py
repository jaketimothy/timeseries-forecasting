import numpy as np
from scipy.stats import binom, bernoulli, norm
from random import choices

class BinomialMarkovSwitchingMultifractal():

    def __init__(self, kbar, m0, sigmabar, b, gamma_kbar):

        self.__kbar = int(kbar)
        assert self.__kbar > 0, "kbar must be positive"
        self.__m0 = float(m0)
        assert self.__m0 >= 1 and self.__m0 <= 2, "m0 must be between 1 and 2"
        self.__sigmabar = float(sigmabar)
        assert self.__sigmabar > 0, "sigma bar must be positive"
        self.__b = float(b)
        assert self.__b > 1, "b must be bigger than 1"
        self.__gamma_kbar = float(gamma_kbar)
        assert self.__gamma_kbar >= 0 and self.__gamma_kbar < 1, "gamma_kbar must be between 0 and 1"

        self.__gamma = self.gamma()

        self.__M = bernoulli(0.5)

    def gamma(self):

        b = self.__b
        gamma_kbar = self.__gamma_kbar
        kbar = self.__kbar

        gamma1 = 1 - (1 - gamma_kbar)**(1 / (b**(kbar-1)))

        gamma = np.array([1 - (1 - gamma1)**(b**k) for k in range(kbar)])

        return gamma

    def sample_M(self):
        m1 = 2 - self.__m0
        return m1 if self.__M.rvs() else self.__m0

    def Mt_update(self, Mt):
        kbar = self.__kbar
        gamma = self.__gamma

        return [choices([self.sample_M(), Mt[k]], weights=[gamma[k], 1-gamma[k]])[0] for k in range(kbar)]

    def r(self, Mt):
        return self.__sigmabar * np.sqrt(np.prod(Mt)) * norm.rvs()

    def simulate(self, n):

        kbar = self.__kbar
        gamma = self.__gamma
        M = self.__M

        M_set = [[self.sample_M() for i in range(kbar)]]
        r_set = [self.r(M_set[0])]
        for t in range(n-1):
            M0 = M_set[-1]
            M_set.append(self.Mt_update(M0))
            r_set.append(self.r(M_set[-1]))

        return (np.array(r_set), np.array(M_set))
