import optuna
import matplotlib.pyplot as plt
from optuna.trial import TrialState
import scipy
import numpy as np
from networks.convLSTM import Seq2Seq

def trials_below(study, threshold=1.0):
    """Count the number of trials with loss (objective value) below the specified threshold."""

    trials_below = optuna.create_study()
    count = 0
    for trial in study.trials:
        if trial.value is not None and trial.value < threshold:
            trials_below.add_trial(trial)
    return trials_below

def postprocessing(study):

    complete_trials = optuna.create_study()
    failed_trials = optuna.create_study()
    pruned_trials = optuna.create_study()
    pruned_trials.add_trials(study.get_trials(deepcopy=False, states=[TrialState.PRUNED]))

    for trial in study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]):
        if trial.value is not None and trial.value < 1:
            complete_trials.add_trial(trial)
        else:
            failed_trials.add_trial(trial)

    return complete_trials, failed_trials, pruned_trials

def plot_num_params(study):


    values = []
    num_params = []
    for i,trial in enumerate(study.trials):
        params = trial.params
        init_features = params["init_features"]
        enc_depth = params["enc_depth"]
        dec_depth = params["dec_depth"]

        enc_conv_features = np.array([init_features, *[64 for _ in range(enc_depth-1)]])
        dec_conv_features = [64 for _ in range(dec_depth)]

        kernel_size = params["kernel_size"]
        enc_kernel_sizes = [kernel_size for _ in range(enc_depth)]
        dec_kernel_sizes = [kernel_size for _ in range(dec_depth)]
        activation = "relu"
        try:
            prev_boxes = params["prev_boxes"]
        except:
            prev_boxes = 1

        model = Seq2Seq(num_channels=3, frame_size=(64,64), prev_boxes=prev_boxes, 
                                extend=2, 
                                num_layers=params["num_layers"],
                                enc_conv_features=enc_conv_features,
                                enc_kernel_sizes=enc_kernel_sizes,
                                dec_conv_features=dec_conv_features,
                                dec_kernel_sizes=dec_kernel_sizes,
                                activation=activation).float()
        
        if trial.value is not None:
            num_params.append(sum(p.numel() for p in model.parameters()))
            values.append(trial.value)

    xy = list(zip(num_params, values))

    xy.sort()

    num_params, values = zip(*xy)

    plt.plot(num_params,values)
    plt.savefig("num_params.png")


# Specify the study name and database URL
study_name = "small_batch"  # Name of the study
storage_url = "sqlite:///{}.db".format(study_name)  # URL of the SQLite database
study_small_batch = optuna.load_study(study_name=study_name, storage=storage_url)

study_name = "second"  # Name of the study
storage_url = "sqlite:///{}.db".format(study_name)
first_study = optuna.load_study(study_name=study_name, storage=storage_url)

with_metrics = optuna.load_study(study_name="with_metrics", storage="sqlite:///with_metrics.db")

study_name = "layers_prev_extend"  # Name of the study
storage_url = "sqlite:///{}.db".format(study_name)
layers_prev_extend = optuna.load_study(study_name=study_name, storage=storage_url)

study_name = "kernel_size_evolution"  # Name of the study
storage_url = "sqlite:///{}.db".format(study_name)
kernel_size_evolution = optuna.load_study(study_name=study_name, storage=storage_url)

combined_study = optuna.create_study()
combined_study.add_trials(study_small_batch.trials)
combined_study.add_trials(first_study.trials)
combined_study.add_trials(with_metrics.trials)
combined_study.add_trials(kernel_size_evolution.trials)

study = optuna.create_study()
for trial in combined_study.trials:
    if trial.value is not None and trial.value < 1:
        study.add_trial(trial)


#fig = optuna.visualization.plot_contour(study, params=["batch_size", "kernel_size","init_features","enc_depth", "dec_depth"])
#fig = optuna.visualization.plot_rank(combined_study, params=["dec_depth", "enc_depth", "init_features", "kernel_size"])
#fig = optuna.visualization.plot_param_importances(combined_study)
#fig = optuna.visualization.plot_slice(layers_prev_extend)
fig = optuna.visualization.plot_slice(study)
#fig = optuna.visualization.plot_timeline(study_small_batch)
fig.show()

