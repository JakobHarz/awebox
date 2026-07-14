import sys
sys.path.extend(['/Users/jakobharzer/MyDrive/Uni/Research/awebox']) # so we can also run this in the console, pycharm does this automatically

from examples.SAM_MPC_experiment_DUALKITE import run_SAM_MPC_experiment_dualkite

# pilot run: sanity-check the dual-kite build/optimize/reconstruct/MPC/export pipeline end-to-end
# d = 2
# N_list = [5]

# target run for Figure 5 replication
d = 3
N_list = [10]

for N in N_list:
    export_dict = run_SAM_MPC_experiment_dualkite(d, N)
