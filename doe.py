import numpy as np
from scipy.stats import qmc
from pyDOE2 import ccdesign
from pyDOE2 import bbdesign

# initial DOE

def get_DOE(doe, dim, population, replicates, function, bounds, rand_noise):
    if doe == 'lhs':
        sampler = qmc.LatinHypercube(dim)
        x_init = np.zeros((replicates, population, dim))
        y_init = np.zeros((replicates, population))
        min_init = np.zeros(replicates)
        for r in range(replicates): 
            x_init_r = sampler.random(n=population)
            x_init_r = qmc.scale(x_init_r, bounds[:,0], bounds[:,1])
            y_init_r = np.array([function(x, rand_noise) for x in x_init_r])
            min_init_r = np.min(y_init_r)
            x_init[r] = x_init_r
            y_init[r] = y_init_r
            min_init[r] = min_init_r

    else: 
        if doe == 'ccd':
            sampler = ccdesign(dim, alpha="r")
        elif doe == 'bbd':
            sampler = bbdesign(dim)
            add = population-sampler.shape[0]
            # BBD design smaller than CCD design, supplement additional points with LHS
            if add > 0:
                sampler = np.vstack((sampler, qmc.scale(qmc.LatinHypercube(dim).random(n=add), [-1]*dim, [1]*dim))) 
        x_init = np.zeros((replicates,population,dim))
        y_init = np.zeros((replicates,population))
        min_init = np.zeros(replicates)

        # scale x_init values by multiplier to bound limits (bound limits are symmetrical)
        multiplier = bounds[0][1]/np.max(sampler)
        for r in range(replicates):
            x_init_r = sampler
            x_init_r = x_init_r*multiplier
            y_init_r = np.array([function(x, noise=rand_noise) for x in x_init_r])
            min_init_r = np.min(y_init_r)
            x_init[r] = x_init_r
            y_init[r] = y_init_r
            min_init[r] = min_init_r

    return x_init, y_init, min_init 