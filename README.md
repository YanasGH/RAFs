# Toward Robust Uncertainty Estimation with Random Activation Functions

This is the code used for performing the experiments from our paper, called "Toward Robust Uncertainty Estimation with Random Activation Functions". Our proposed method, the _Random Activation Functions (RAFs) Ensemble_, is evaluated against five popular baselines (i) Deep Ensemble (DE) [1], Neural Tangent Kernel Gaussian Process Parameter Ensemble (NTKGP-param) [2], Anchored Ensemble (AE) [3], Bootstrapped Ensemble of NNs Coupled with Random Priors (RP-param) [4], and Hyperdeep Ensemble (HDE) [5]. The experiments are described and discussed in detail in the paper.


## A. Requirements

* Python>=3.9

## B. Dependencies

First prepare and activate your python environment (e.g. `conda`). Then, to perform the regression experiments use the requirments.txt from the folder regression.  
Using pip:
```bash
$ pip install -r requirements.txt

```

## C. Main Regression Experiments

### How to run 

First navigate to the folder _main_experiments_. Then run the following:

```bash
$ python run_experiments.py

```
This will evaluate all methods on the He dataset as it is set as a default. However, you can choose any of the provided datasets by running the following line ("Robot arm" used as an example):

```bash
$ python run_experiments.py --dataset "Robot arm"

```
All available datasets are:
- "He"
- "Forrester"
- "Schaffer"
- "Double pendulum" 
- "Rastrigin"
- "Ishigami" 
- "Environmental model"
- "Griewank"
- "Roos & Arnold"
- "Friedman"
- "Planar arm torque"
- "Sum of powers"
- "Ackley"
- "Piston simulation" 
- "Robot arm"
- "Borehole"
- "Styblinski-Tang"
- "PUMA560"
- "Adapted Welch"
- "Wing weight"
- "Boston housing"
- "Abalone"
- "Naval propulsion plant"
- "Forest fire"
- "Parkinson"

#### C.1 To save *Figure 1* from the paper
First navigate to the bottom of all methods' py files (i.e. ae, de, hde, ntkgpparam, rpparam, rafs). There the function `viz_one_d` is used to visualize the performance. Change its last argument to "True" (as a default it is set to "False"). Then execute the experiment by running:

```bash
$ python run_experiments.py

```
Similarly, the techniques' performance on the other one-dimensional dataset, that is "Forrester", can be displayed. That can be done by following the same procedure, but running the following instead:

```bash
$ python run_experiments.py --dataset "Forrester"

```

## D. Additional Regression Experiments

### How to run
First navigate to the folder _additional_experiments_.

### D1. Scalability
RAFs Ensemble is evaluated against RP-param on two larger high-dimensional datasets, such that both methods utilize a more complex architecture (that is two hidden layers of 128 neurons). The datasets of choice are "Superconductivity" and "Popularity" (65D and 40D resp.). To conduct this experiment, run the following line:

```bash
$ python run_additional_experiments.py

```
This will output the results for the "Superconductivity" dataset, as it is set as a default. To test both methods on the "Popularity" dataset, please run:

```bash
$ python run_additional_experiments.py --dataset "Popularity"

```

### D.1 RAFs methodology applied on RP-param
To observe the positive effect of RAFs methodology on RP-param on the Parkinson's dataset, run the following:

```bash
$ python run_rpparam_combined_with_rafs.py

```
**NB:** This experiment is conducted only on the Parkinson's dataset.



## E. References 

[1] Lakshminarayanan, B.; Pritzel, A.; and Blundell, C. 2017. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.

[2] He, B.; Lakshminarayanan, B.; and Teh, Y. W. 2020. Bayesian Deep Ensembles via the Neural Tangent Kernel.

[3] Pearce, T.; Zaki, M.; Brintrup, A.; and Neely, A. D. 2018. Uncertainty in Neural Networks: Bayesian Ensembling.

[4] Osband, I.; Aslanides, J.; and Cassirer, A. 2018. Randomized Prior Functions for Deep Reinforcement Learning.

[4] Wenzel, F.; Snoek, J.; Tran, D.; and Jenatton, R. 2020. Hyperparameter Ensembles for Robustness and Uncertainty Quantification.
	
## F. Contact

Maintained by Yana Stoyanova (YanasGH).
