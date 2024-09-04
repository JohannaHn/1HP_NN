import optuna
import matplotlib
from optuna.trial import TrialState
import scipy
import numpy as np



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


# Specify the study name and database URL
study_name = "small_batch"  # Name of the study
storage_url = "sqlite:///{}.db".format(study_name)  # URL of the SQLite database
study_small_batch = optuna.load_study(study_name=study_name, storage=storage_url)

study_name = "second"  # Name of the study
storage_url = "sqlite:///{}.db".format(study_name)
first_study = optuna.load_study(study_name=study_name, storage=storage_url)

trials_below_one = trials_below(first_study, 1)

jitter = np.random.rand(len(trials_below_one.trials) + len(study_small_batch.trials)) - 0.5

combined_study = optuna.create_study()
combined_study.add_trials(trials_below_one.trials)
combined_study.add_trials(study_small_batch.trials)

transformed_study = optuna.create_study()

for i,trial in enumerate(trials_below_one.trials):
    trial.set_user_attr("dec_depth_usr", trial.params["dec_depth"] + jitter[i])
    transformed_study.add_trial(trial)

for i,trial in enumerate(study_small_batch.trials):
    trial.set_user_attr("dec_depth_usr", trial.params["dec_depth"] + jitter[i])
    transformed_study.add_trial(trial)


#fig = optuna.visualization.plot_contour(study, params=["batch_size", "kernel_size","init_features","enc_depth", "dec_depth"])
#fig = optuna.visualization.plot_rank(combined_study, params=["dec_depth", "enc_depth", "init_features", "kernel_size"])
fig = optuna.visualization.plot_param_importances(combined_study)
#fig = optuna.visualization.plot_slice(combined_study)
fig.show()
