# basic packages
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from scipy.stats import qmc
from pyDOE2 import ccdesign

# self-defined utils
from test_function import test_function
from doe import get_DOE
from optimizers import optimize

# Problem constants ======================================================================================================================
dim = 5                                          # 5, 20 or 40
population = 50                                  # for dim=5, will be ignored if CCD or BBD design method is chosen
iteration = 10                                   # number of iterations
offset=0.4                                       # offset value for function input
noisy = False                                    # if to add noise to function
noise_ratio = 0.2                                # var for gaussian noise, range [0 - 1]. y = y + N(0, y*noise)
replicates = 10                                  # number of replicates to run
test_set = range(1, 25)                          # iterable. Specifiy test function by id 1-24. If test all: range(1,25), if only 1: [1]
methods = ['GA',                                 # methods
           'DE',
           'PSO',
           '2OP-PV-GA',
           'KRG-EI-GA',
           'KRG-PV-GA',
           'MLP-PV-GA',
           'SVM-PV-GA',
           'KRG-truncGA PV',
           'KRG-truncDE PV',
           'KRG-L-BFGS-B PV']

doe_list = ['lhs', 'bbd', 'ccd']                 # DOE methods. Only relevant for dim=5, for dim=20 or 40, only LHS will be used
# Unit test for methods
# methods = [methods[4]]

# Problem variables ======================================================================================================================
assert dim in (5, 20, 40), "Invalid dimension!"
if dim==5:
    save_folder = "1.low dim results "
    if 'ccd' in doe_list:
        population = ccdesign(dim).shape[0]
    elif 'bbd' in doe_list:
        population = bbdesign(dim).shape[0]
elif dim==20:
    save_folder = "2.mid dim results "
    doe_list = ['lhs']    
else:
    save_folder = "3.high dim results "
    doe_list = ['lhs']   

if noisy:
    save_folder += "(w noise)"
else:
    save_folder += "(no noise)"

# Run experiments ========================================================================================================================
for doe in doe_list:
    for function_id in test_set:
        
        # dataframe for storing results for plotting
        results = pd.DataFrame(columns=["value", "iteration", "algorithm", "replicate"])
        result_path = os.path.join(os.getcwd(), save_folder, f'f{function_id}_{dim}-d_{doe}.csv')


        # get function and initial DOE
        function, bounds, rand_noise, problem = test_function(dim, function_id, noisy, noise_ratio, offset)
        x_init, y_init, min_init  = get_DOE(doe, dim, population, replicates, function, bounds, rand_noise)


        # perform optimization
        results = pd.DataFrame(columns=["value", "iteration", "algorithm", "replicate"])
        result_path = os.path.join(os.getcwd(), save_folder, f'f{function_id}_{dim}-d_{doe}.csv')

        for method in methods:
            results = pd.concat((results, optimize(method, replicates, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds)))
            results = results.reset_index(drop=True)
            results.to_csv(result_path)


        # create and save plot
        sns.set_theme()
        sns.set(rc={'figure.figsize':(10, 7.5)})
        sns.lineplot(data=results, x="iteration", y="value", hue="algorithm")
        image_path = os.path.join(os.getcwd(), save_folder, f'f{function_id}_{dim}-d_{doe}.png')
        plt.savefig(image_path, dpi=100)


        # clear objects
        plt.clf()
        problem.free()
