from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import scipy

# optimizers
import pygad
import pyswarms as ps
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from smt.surrogate_models import KRG
from smt.applications import EGO
from scipy_mod import de_generator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

def GA(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    funval_i_GA = []
    def callback_gen(ga_instance):
        sol=ga_instance.best_solution()[0]
        funval_i_GA.append(function(sol, noise=0))

   
    ga_instance = pygad.GA(num_generations=iteration,
                            num_parents_mating=population,
                            fitness_func=lambda x, y: -function(x,noise=rand_noise), 
                            initial_population=x_init[r],
                            init_range_low=bounds[:,0][0],
                            init_range_high=bounds[:,1][0],
                            mutation_num_genes=1,
                            callback_generation=callback_gen) 
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    n = len(funval_i_GA)
    if n < iteration:
        for i in range(iteration-n):
            funval_i_GA.append(funval_i_GA[-1])
           
    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_GA), 0, min_init[r])
    funval_i_GA = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_GA[i] = y_min
    return funval_i_GA


def DE(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    funval_i_DE = []
    def callback(xk, convergence):
        funval_i_DE.append(function(xk, noise=0))

    de = differential_evolution(function, bounds, args=(rand_noise,), init=x_init[r].copy(), popsize=population//dim, maxiter=iteration, polish=False, callback=callback)
   
    # if terminate early, copy last value to all subsequent incomplete iterations
    n = len(funval_i_DE)
    if n < iteration:
        for i in range(iteration-n):
            last = funval_i_DE[-1]
            funval_i_DE.append(last)

    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_DE), 0, min_init[r])
    funval_i_DE = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_DE[i] = y_min
    return funval_i_DE


def PSO(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    max_bound = (bounds[:,1]+0.000000000000001) * np.ones(dim)
    min_bound = - max_bound
    bound = (min_bound, max_bound)
    
    def functionpso(x,noise=rand_noise):
        y=[]
        for i in x:
            y.append(function(i,noise))
        return y
    
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    funval_i_PSO = []
    # Call instance of PSO with bounds argument
    optimizer = ps.single.GlobalBestPSO(n_particles=population, dimensions=dim, options=options, bounds=bound, init_pos=x_init[r])

    # Perform optimization
    cost, pos = optimizer.optimize(functionpso, iters=iteration+1, verbose=False)
    pos_history=optimizer.pos_history
    funval_i_PSO = [min(functionpso(x,noise=0)) for x in pos_history[1:]]
    funval_i_PSO = np.insert(np.array(funval_i_PSO), 0, min_init[r])
    n = len(funval_i_PSO)
    if n < iteration: 
        for i in range(iteration-n):
            funval_i_PSO.append(funval_i_PSO[-1])

    # ensure convergence plot does not increase
    y_i = funval_i_PSO
    funval_i_PSO = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_PSO[i] = y_min

    return funval_i_PSO


def RSM_PV_GA(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    n = population # top n to add each iteration to fit model
   
    def fitness_func(x,y):
        x=x.reshape(-1,dim)
        return -model.predict(x)[0]

    x_t = x_init[r].copy()
    y_t = y_init[r].copy()
   
    # outer loop
    funval_2OP = np.min(y_t)
    funval_i_2OP = []
    bar = tqdm(range(iteration))  #tqdm is for the progress bar 
    for i in bar:
        # train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poly = PolynomialFeatures(degree=2)
            poly_x_values = poly.fit_transform(x_t)
            model = Pipeline([('poly',PolynomialFeatures(degree=2)),
                              ('linear',LinearRegression(fit_intercept=False))])
            model = model.fit(x_t, y_t)
            model.named_steps['linear'].coef_
           
           
            # inner loop: run GA on surrogate model
            ga_instance = pygad.GA(num_generations=100, #number of iterations can be infinite, you're only testing the sm, not the actual blackbox function that needs to be checked experimentally
                                    num_parents_mating=population,
                                    fitness_func=fitness_func,
                                    initial_population=x_t,
                                    init_range_low=bounds[:,0][0],
                                    init_range_high=bounds[:,1][0]) 
        ga_instance.run()
        pop = ga_instance.population
        y_pop = np.array([function(x, rand_noise) for x in pop.reshape(-1, dim)])
        pop_ind = np.argsort(y_pop)
        x_t = np.vstack((x_t, pop[pop_ind][:n]))
        y_t = np.append(y_t, y_pop[pop_ind][:n])
        
        b_ind = np.argmin(y_t)
        if y_t[b_ind] < funval_2OP:
            x_2OP = x_t[b_ind]
            funval_2OP = function(x_2OP, noise=0)
        funval_i_2OP.append(funval_2OP)
        bar.set_description(f"replicate {r}, cur best: {funval_2OP:.2f}")
   
    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_2OP), 0, min_init[r])
    funval_i_2OP = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_2OP[i] = y_min

    return funval_i_2OP


def KRG_EI_GA(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    def fun_EGO(x):
        if x.ndim == 1:
            return function(x, rand_noise)
        return np.array([function(x_i, rand_noise) for x_i in x])

    # #### SBO-EI: add n highest EI each iteration

    n = population # top n to add each iteration to fit model
    x_t = x_init[r].copy()
    y_t = y_init[r].copy()    
   
    # outer loop
    funval_EI = min_init[r]
    funval_i_EI = []
    bar = tqdm(range(iteration))   
    for i in bar:    
        # train ego manually
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # initialize ego object
            if noisy:
                #sm = KRG(theta0=[1e-2]*dim, eval_noise=True, print_global=False)
                sm = KRG(theta0=[1e-2]*dim, eval_noise=True, use_het_noise=False, print_global=False)
                ego = EGO(n_iter=1, criterion='EI', xdoe=x_t, xlimits=bounds, surrogate=sm)
            else:
                ego = EGO(n_iter=1, criterion='EI', xdoe=x_t, xlimits=bounds)
            x_opt, y_opt, _, _, _ = ego.optimize(fun=fun_EGO)
            ego.gpr.set_training_values(x_t, y_t)
            ego.gpr.train()

            # inner loop: run DE on surrogate model
            def EI_fun(x,y):
                ei = ego.EI(x.reshape(-1, dim))
                if type(ei) == np.ndarray:
                    return ei.reshape(-1)[0]
                else:
                    return ei
           
            # inner loop: run GA on surrogate model
            ga_instance = pygad.GA(num_generations=100, #number of iterations can be infinite, you're only testing the sm, not the actual blackbox function that needs to be checked experimentally
                                    num_parents_mating=population,
                                    fitness_func=EI_fun,
                                    initial_population=x_t,
                                    init_range_low=bounds[:,0][0],
                                    init_range_high=bounds[:,1][0]) 
        ga_instance.run()
        pop = ga_instance.population
        y_pop = np.array([function(x, rand_noise) for x in pop.reshape(-1, dim)])
        pop_ind = np.argsort(y_pop)
        x_t = np.vstack((x_t, pop[pop_ind][:n]))
        y_t = np.append(y_t, y_pop[pop_ind][:n])
        
        b_ind = np.argmin(y_t)
        if y_t[b_ind] < funval_EI:
            x_EI = x_t[b_ind]
            funval_EI = function(x_EI, noise=0)
        funval_i_EI.append(funval_EI)
        bar.set_description(f"replicate {r}, cur best: {funval_EI:.2f}")

    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_EI), 0, min_init[r])
    funval_i_EI = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_EI[i] = y_min

    return funval_i_EI


def KRG_PV_GA(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    n = population # top n to add each iteration to fit model

    def fitness_func(x,y):
        x=x.reshape(-1,dim)    
        return -sm.predict_values(x)[0][0]

    x_t = x_init[r].copy()
    y_t = y_init[r].copy()

    # outer loop
    funval_KRG = np.min(y_t)
    funval_i_KRG = []
    bar = tqdm(range(iteration))
    first=0
    for i in bar:
        # train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = KRG(theta0=[1e-2]*dim, print_global=False)
            sm.set_training_values(x_t, y_t)
            sm.train()

            # inner loop: run GA on surrogate model
            ga_instance = pygad.GA(num_generations=100, #number of iterations can be infinite, you're only testing the sm, not the actual blackbox function that needs to be checked experimentally
                                   num_parents_mating=population,
                                   fitness_func=fitness_func,
                                   initial_population=x_t,
                                   init_range_low=bounds[:,0][0],
                                   init_range_high=bounds[:,1][0]) 
        ga_instance.run()
        pop = ga_instance.population
        y_pop = np.array([function(x, rand_noise) for x in pop.reshape(-1, dim)])
        pop_ind = np.argsort(y_pop)
        x_t = np.vstack((x_t, pop[pop_ind][:n]))
        y_t = np.append(y_t, y_pop[pop_ind][:n])
        
        b_ind = np.argmin(y_t)
        if y_t[b_ind] < funval_KRG:
            x_KRG = x_t[b_ind]
            funval_KRG = function(x_KRG, noise=0)
        funval_i_KRG.append(funval_KRG)
        bar.set_description(f"replicate {r}, cur best: {funval_KRG:.2f}")
        
    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_KRG), 0, min_init[r])
    funval_i_KRG = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_KRG[i] = y_min

    return funval_i_KRG


def MLP_PV_GA(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    n = population # top n to add each iteration to fit model
         
    def fitness_func(x,y):
        x=x.reshape(-1,dim)    
        return -sm.predict(x)[0]

    x_t = x_init[r].copy()
    y_t = y_init[r].copy()
     
    # outer loop
    funval_MLP = np.min(y_t)
    funval_i_MLP = []
    bar = tqdm(range(iteration))  #tqdm is for the progress bar 
    for i in bar:
        # train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = MLPRegressor(early_stopping=False)
            sm.fit(x_t, y_t)
 
             
            # inner loop: run GA on surrogate model
            ga_instance = pygad.GA(num_generations=100, #number of iterations can be infinite, you're only testing the sm, not the actual blackbox function that needs to be checked experimentally
                                   num_parents_mating=population,
                                   fitness_func=fitness_func,
                                   initial_population=x_t,
                                   init_range_low=bounds[:,0][0],
                                   init_range_high=bounds[:,1][0]) 
        ga_instance.run()
        pop = ga_instance.population
        y_pop = np.array([function(x, rand_noise) for x in pop.reshape(-1, dim)])
        pop_ind = np.argsort(y_pop)
        x_t = np.vstack((x_t, pop[pop_ind][:n]))
        y_t = np.append(y_t, y_pop[pop_ind][:n])
        
        b_ind = np.argmin(y_t)
        if y_t[b_ind] < funval_MLP:
            x_MLP = x_t[b_ind]
            funval_MLP = function(x_MLP, noise=0)
        funval_i_MLP.append(funval_MLP)
        bar.set_description(f"replicate {r}, cur best: {funval_MLP:.2f}")
     
    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_MLP), 0, min_init[r])
    funval_i_MLP = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_MLP[i] = y_min

    return funval_i_MLP


def SVM_PV_GA(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    n = population # top n to add each iteration to fit model

    def fitness_func(x,y):
        x=x.reshape(-1,dim)    
        return -sm.predict(x)[0]

    x_t = x_init[r].copy()
    y_t = y_init[r].copy()
    
    # outer loop
    funval_SVR = np.min(y_t)
    funval_i_SVR = []
    bar = tqdm(range(iteration))  #tqdm is for the progress bar 
    for i in bar:
        # train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = SVR()
            sm.fit(x_t, y_t)
       
            # inner loop: run GA on surrogate model
            ga_instance = pygad.GA(num_generations=100, #number of iterations can be infinite, you're only testing the sm, not the actual blackbox function that needs to be checked experimentally
                                   num_parents_mating=population,
                                   fitness_func=fitness_func,
                                   initial_population=x_t,
                                   init_range_low=bounds[:,0][0],
                                   init_range_high=bounds[:,1][0]) 
        ga_instance.run()
        pop = ga_instance.population
        y_pop = np.array([function(x, rand_noise) for x in pop.reshape(-1, dim)])
        pop_ind = np.argsort(y_pop)
        x_t = np.vstack((x_t, pop[pop_ind][:n]))
        y_t = np.append(y_t, y_pop[pop_ind][:n])
        
        b_ind = np.argmin(y_t)
        if y_t[b_ind] < funval_SVR:
            x_SVR = x_t[b_ind]
            funval_SVR = function(x_SVR, noise=0)
        funval_i_SVR.append(funval_SVR)
        bar.set_description(f"replicate {r}, cur best: {funval_SVR:.2f}")
    
    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_SVR), 0, min_init[r])
    funval_i_SVR = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_SVR[i] = y_min

    return funval_i_SVR


def KRG_PV_truncGA(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    n = population
    x_t = x_init[r].copy()
    y_t = y_init[r].copy()  

    def fitness_fun(x,y):
            return -sm.predict_values(x.reshape(-1,dim))[0][0]
    # outer loop
    funval_truncGA = min_init[r]
    funval_i_truncGA = []
    bar = tqdm(range(iteration))   
    for i in bar:
        # train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = KRG(theta0=[1e-2]*dim, print_global=False)
            sm.set_training_values(x_t, y_t)
            sm.train()
           
            # inner loop: run GA on surrogate model
            ga_instance = pygad.GA(num_generations=10, #number of iterations can be infinite, you're only testing the sm, not the actual blackbox function that needs to be checked experimentally
                                    num_parents_mating=population,
                                    fitness_func=fitness_fun,
                                    initial_population=x_t,
                                    init_range_low=bounds[:,0][0],
                                    init_range_high=bounds[:,1][0]) 
        ga_instance.run()
        pop = ga_instance.population
        y_pop = np.array([function(x, rand_noise) for x in pop.reshape(-1, dim)])
        pop_ind = np.argsort(y_pop)
        x_t = np.vstack((x_t, pop[pop_ind][:n]))
        y_t = np.append(y_t, y_pop[pop_ind][:n])
        
        b_ind = np.argmin(y_t)
        if y_t[b_ind] < funval_truncGA:
            x_truncGA = x_t[b_ind]
            funval_truncGA = function(x_truncGA, noise=0)
        funval_i_truncGA.append(funval_truncGA)
        bar.set_description(f"replicate {r}, cur best: {funval_truncGA:.2f}")

    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_truncGA), 0, min_init[r])
    funval_i_truncGA = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_truncGA[i] = y_min

    return funval_i_truncGA


def KRG_PV_truncDE(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    n = population
    x_t = x_init[r].copy()
    y_t = y_init[r].copy()    
    
    # outer loop
    funval_truncDE = np.min(y_t)
    funval_i_truncDE = []
    bar = tqdm(range(iteration))   
    for i in bar:
        # train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = KRG(theta0=[1e-2]*dim, print_global=False)
            sm.set_training_values(x_t, y_t)
            sm.train()

        # inner loop: run DE on surrogate model
        ret, pop, _ = de_generator(lambda x: sm.predict_values(x.reshape(-1,dim)), bounds, init=x_t, strategy='rand1bin', maxiter=10)
        y_pop = np.array([function(x, rand_noise) for x in pop.reshape(-1, dim)])
        pop_ind = np.argsort(y_pop)
        x_t = np.vstack((x_t, pop[pop_ind][:n]))
        y_t = np.append(y_t, y_pop[pop_ind][:n])

        b_ind = np.argmin(y_t)
        if y_t[b_ind] < funval_truncDE:
            x_truncDE = x_t[b_ind]
            funval_truncDE = function(x_truncDE, noise=0)
        funval_i_truncDE.append(funval_truncDE)
        bar.set_description(f"replicate {r}, cur best: {funval_truncDE:.2f}, DE iters: {ret.nit}")

    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_truncDE), 0, min_init[r])


    funval_i_truncDE = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_truncDE[i] = y_min

    return funval_i_truncDE


def KRG_PV_LBFGS(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    n = population
    x_t = x_init[r].copy()
    y_t = y_init[r].copy()   

    # outer loop
    funval_BFGS = np.min(y_t)
    funval_i_BFGS = []
    bar = tqdm(range(iteration))   
    for i in bar:
        # train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = KRG(theta0=[1e-2]*dim, print_global=False)
            sm.set_training_values(x_t, y_t)
            sm.train()

        # inner loop: run L-BFGS-B on surrogate model
        pop = np.zeros((len(x_t),dim))
        for i in range(len(x_t)):
            result = minimize(lambda x: sm.predict_values(x.reshape(-1,dim)), x_t[i], method='L-BFGS-B')
            pop[i]= result['x']
        y_pop = np.array([function(x, rand_noise) for x in pop.reshape(-1, dim)])
        pop_ind = np.argsort(y_pop)
        x_t = np.vstack((x_t, pop[pop_ind][:n]))
        y_t = np.append(y_t, y_pop[pop_ind][:n])

        b_ind = np.argmin(y_t)
        if y_t[b_ind] < funval_BFGS:
            x_BFGS = x_t[b_ind]
            funval_BFGS = function(x_BFGS, noise=0)
        funval_i_BFGS.append(funval_BFGS)
        bar.set_description(f"replicate {r}, cur best: {funval_BFGS:.2f}")

    # ensure convergence plot does not increase
    y_i = np.insert(np.array(funval_i_BFGS), 0, min_init[r])
    funval_i_BFGS = np.zeros_like(y_i)
    y_min = y_i[0]
    for i, funval in enumerate(y_i):
        if funval < y_min:
            y_min = funval
        funval_i_BFGS[i] = y_min   

    return funval_i_BFGS


def optimize(method, replicates, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    if method == 'GA':
        optimizer = GA
    elif method == 'DE':
        optimizer = DE
    elif method == 'PSO':
        optimizer = PSO
    elif method == '2OP-PV-GA':
        optimizer = RSM_PV_GA
    elif method == 'KRG-EI-GA':
        optimizer = KRG_EI_GA
    elif method == 'KRG-PV-GA':
        optimizer = KRG_PV_GA
    elif method == 'MLP-PV-GA':
        optimizer = MLP_PV_GA
    elif method == 'SVM-PV-GA':
        optimizer = SVM_PV_GA
    elif method == 'KRG-truncGA PV':
        optimizer = KRG_PV_truncGA
    elif method == 'KRG-truncDE PV':
        optimizer = KRG_PV_truncGA
    elif method == 'KRG-L-BFGS-B PV':
        optimizer = KRG_PV_LBFGS
    else:
        raise ValueError("method not found!")

    repbar = tqdm(range(replicates))
    results = pd.DataFrame(columns=["value", "iteration", "algorithm", "replicate"])

    for r in repbar:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            funval_i = optimizer(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds)
        results = pd.concat((results, pd.DataFrame({'value': funval_i, 'iteration': np.arange(iteration+1), 'algorithm': f'{method}', 'replicate': int(r)})))
    
    results = results.reset_index(drop=True)

    return results
