import numpy as np
from scipy.stats import qmc

# test functions
import cocopp
import BBOB_test_functions
from BBOB_test_functions import BBOB_function

def test_function(dim, function_id, noisy, noise_ratio, offset):
    # define test function
    print("****************************************")
    problem, bounds = BBOB_function(dimensions=dim, function_id=function_id)
    print("****************************************\n")
    bounds = np.array(bounds)

    # define additive noise amount
    if not noisy: 
        rand_noise = 0
    else:
        sampler = qmc.LatinHypercube(dim, seed=42)
        x_rand_sample = sampler.random(n=1000)
        x_rand_sample = qmc.scale(x_rand_sample, bounds[:,0], bounds[:,1])
        rand_noise = noise_ratio * np.std(np.array([problem(x) for x in x_rand_sample]))    

    def function(x, noise=0, offset=offset):
        assert x.ndim == 1, "x must be single vector"
        y = problem(x-offset)
        #return y + np.random.normal(0, np.abs(y)*noise)
        return y + np.random.normal(0, noise)

    return function, bounds, rand_noise, problem