# Review-of-culture-media-optimization-methods
Benchmark comparison experiments for the paper "A review of algorithmic approaches for cell culture media optimization"

## Installation
1. Create conda environment
```
conda env create -f environment.yml
```
2. Follow instructions to install COCO (for test functions) provided here: https://github.com/numbbo/coco 

## Run experiments
1. Edit the problem constants in `run_experiment.py` to run the experiment of choice
- **dim** : dimension of problem, choices are 5, 20, or 40
- **population** : population size of each generation. For dim=5, this will be ignored if CCD or BBD DOE is chosen, and replaced by the default number as determined by the respective DOE methods
- **iteration** : number of iterations
- **offset** : offset value for function input, i.e. f(x) -> g(x - offset), where g is the original BBOB test function. Purpose of offset is prevent the minima solutions at x = 0, which is present in standard DOEs
- **noisy** : if to add noise to function
- **noise_ratio** : controls variance for gaussian noise, range [0 - 1]. y = y + N(0, y*noise)
- **replicates** : number of replicates to perform per experiment
- **test_set** : an interable to specify test function by ID (1-24). E.g. If test all: range(1,25), if only 1: [1]
- **methods** : list of all methods used in this experiment. If a subset of methods is to be used, override this list with a custom list

2. In the `optimizer_exp` conda environment, run script:
```
python run_experiment.py
```
3. Find results stored in corresponding results folder
