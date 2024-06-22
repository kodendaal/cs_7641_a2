# CS7641 Assignment 2: Random Optimization

Kirsten Odendaal (24/06/24)

REPO URL: https://github.com/kodendaal/cs_7641_a2.git

Please clone this git to a local project folder if you would like to replicate the experiments in the assignment

```git clone https://github.com/kodendaal/cs_7641_a2.git```

Requirements:
----
This file contains all the necessary packages for this project.

* Ensure that Python 3.11 is installed
* Ensure that pip and git are installed
* Includes modified pyperch repo for neural network assessments
* Running ```pip install -r requirements.txt``` will install all the packages in your project's environment

Datasets:
----
The one datasets used is Wine Classification. It is found in the associated 'datasets' folder. It can also be directly downloaded from its source on Kaggle:

* https://www.kaggle.com/datasets/yasserh/wine-quality-dataset


Files:
----

The 2 Problem Random Optimization evaluation file is contained in a jupyter notebook: ```main.ipynb```. 

The Neural Network Random Optimization evaluation file is contained in a jupyter notebook: ```main_nn.ipynb```. 

Helper evaluation functions are contained in a python file: ```utils.py```.

----
The general running structure is as follows;
RO 2-problem:
1. Select problem you would like to evaluate and load in selection cell. 
    * Indicate whether you would like to load in previous pickle files in outputdir folder, or;
    * perform new_evaluations

For each optimizer (Random Hill, Simulated Annealing, and Genetic Algorithm);
2. Peform preliminary multi-seeded evaluations with varying hyper-parameters. Manually set parameter ranges fo interest. 
3. Save and store evaluated metrics and prints best results 
4. Visualize metrics: Fitness (Hyperparams) vs Iterations, FEvals (Hyperparms) vs Iterations, Fitness (Complexity) vs Iterations, Fitness (Complexity) vs Computation Time, FEvals (Complexity) vs Computation Time
5. Visualzie comparison summary - FEvals/Time and Time (Waterfall plot)

RO Neural Network:
1. Select dataset you would like to evaluate and load in selection cell. 
    * Indicate whether you would like to load in previous pickle files in outputdir folder, or;
    * Perform new gridsearch evaluations, or;
    * Plot the evaluated Loss curves, or;
    * Plot the evalauted Learning curves
2. Load and split data into training and testing set. (Perform data standardization)

For each optimizer (Backpropagation, Random Hill, Simulated Annealing, and Genetic Algorithm);
3. Manually set hyperparameter ranges and evaluate grid-search (store resulting models)
4. Using best params - Re-evaluate on multi-seed and append statistics. Evaluate final model on test set.
5. Visualize neural network loss curves - Training/Validation
6. Re-evalaute neural network model with best epoch results (check test performance)
7. Visualize interlayer weights for convergence comparison study
8. Visualize neural network learning curves (not used in report)
9. Visualzie weights distribition comparison summary (includes chi-squared metric) 


Write-up:
----
Overleaf read-only link: https://www.overleaf.com/read/gygsncfrrmdg#da3c9e
