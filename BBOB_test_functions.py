import numpy as np
import cocoex

fun_bounds = dict(ackley = [(-32.7, 32.7)])

def ackley(x, offset=0, noise=0):
    """
    f(x*) = 0 at x* = (0,...,0)
    Parameters
    ----------
    x : numpy array
    """
    x = x-offset
    y = -20*np.exp(-0.2*np.sqrt(1/len(x)*np.sum(x**2))) - np.exp(1/len(x)*np.sum(np.cos(2*np.pi*x))) + 20 + np.exp(1)
    return y + np.random.normal(0, np.abs(y)*noise)

def BBOB_function(dimensions, function_id, instances=1):
    """
    Returns the function as indicated by function_id from BBOB suite.
    The 1st function instance is returned by default
    Parameters
    ----------
    dimensions : int, must be in [2,3,5,10,20,40]
    function_id: int, >=1 and <= 24
    instances: int, default=1, >= 1 and <= 15
    """
    assert type(dimensions)==int and dimensions in [2, 3, 5, 10, 20, 40], "dimensions must be one of [2,3,5,10,20,40]"
    assert type(function_id)==int and function_id >=1 and function_id <= 24, "function_id must be int >=1 and <=24"
    assert type(instances)==int and instances >=1 and instances <= 15, "instances must be int >=1 and <=15"

    # get problem
    suite = cocoex.Suite("bbob", "instances: 1", f"dimensions: {dimensions} function_indices: {function_id}")
    problem = suite[0]
    print(problem.name)

    # get bounds
    lb = problem.lower_bounds, 
    ub = problem.upper_bounds
    if type(ub) == tuple and len(ub) == 1:
        ub = ub[0]
    if type(lb) == tuple and len(lb) == 1:
        lb = lb[0]
    bounds = list(zip(lb, ub))

    return problem, bounds
