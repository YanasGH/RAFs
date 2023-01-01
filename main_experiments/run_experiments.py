# import libs
import argparse

def main():
    cmdline_parser = argparse.ArgumentParser(description='Script for running ensembles on chosen dataset.')
    cmdline_parser.add_argument('--dataset', help='Choose the dataset from the list: "He", "Forrester", "Schaffer", "Double pendulum", "Rastrigin", "Ishigami", "Environmental model", "Griewank", "Roos & Arnold", "Friedman", "Planar arm torque", "Sum of powers", "Ackley", "Piston simulation", "Robot arm", "Borehole", "Styblinski-Tang", "PUMA560", "Adapted Welch", "Wing weight", "Boston housing", "Abalone", "Naval propulsion plant", "Forest fire", "Parkinson"', default='He')
    args = cmdline_parser.parse_args()
    dname = args.dataset
    return dname
dname = main()     # python run_experiments.py --dataset "He" 

all_datasets = ["He", "Forrester", "Schaffer", "Double pendulum", "Rastrigin", "Ishigami", "Environmental model", "Griewank", "Roos & Arnold", "Friedman", "Planar arm torque", "Sum of powers", "Ackley", "Piston simulation", "Robot arm", "Borehole", "Styblinski-Tang", "PUMA560", "Adapted Welch", "Wing weight", "Boston housing", "Abalone", "Naval propulsion plant", "Forest fire", "Parkinson"]
if dname not in all_datasets:
    print('{} is not implemented. Please choose from: {}'.format(dname, all_datasets))
    
else:
    print("---------- Running all baselines and RAFs Ensemble on the {} dataset: ----------".format(dname))
    print("---------- Anchored Ensemble ----------")
    import ae
    print("---------- RAFs Ensemble ----------")
    import rafs
    print("---------- RP-param ----------")
    import rpparam
    print("---------- Hyperdeep Ensemble ----------")
    import hde
    print("---------- Deep Ensemble ----------")
    import de
    print("---------- NTKGP-param ----------")
    import ntkgpparam