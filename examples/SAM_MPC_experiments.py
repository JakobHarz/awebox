import sys
sys.path.extend(['/Users/jakobharzer/MyDrive/Uni/Research/awebox']) # so we can also run this in the console, pycharm does this automatically

from examples.SAM_MPC_experiment import run_SAM_MPC_experiment

# d = 4 # manually 'parellize' over d
# N_list = [1,2,3,4,5,6,7,10,15]

# d = 5 # manually 'parellize' over d
# N_list = [10,15,20,25,30]

d = 6 # manually 'parellize' over d
N_list = [20,30,40]

for N in N_list:
    export_dict = run_SAM_MPC_experiment(d, N)