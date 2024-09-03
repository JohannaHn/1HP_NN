import optuna
import matplotlib
from optuna.trial import TrialState
import scipy



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
study_name = "second"  # Name of the study
storage_url = "sqlite:///{}.db".format(study_name)  # URL of the SQLite database

study = optuna.load_study(study_name=study_name, storage=storage_url)
complete_trials, failed_trials, pruned_trials = postprocessing(study)

trials_below_one = trials_below(study, 1)

print(f'number of trials: {len(study.trials)}')
print(f'number of pruned trials: {len(pruned_trials.trials)}')
print(f'number of complete trials: {len(complete_trials.trials)}')
print(f'number of trials below 1: {len(trials_below_one.trials)}')

print(study.trials.__getitem__(36).value)
print(study.trials.__getitem__(37).value)

print("Best trial:")
print(study.trials.__getitem__(9).value)
print(study.best_trial.value)
print(study.best_params)
print(study.best_value)

new_study = optuna.create_study()

count = 0
for trial in study.trials:
    if trial.params["kernel_size"] == 9 and trial.params["batch_size"] > 16:
        count += 1
print(f'nr of trials with kernel size 9 and batch > 16: {count}')

for trial in pruned_trials.trials:
    trial.value = 1
    trial.state=TrialState.COMPLETE
    new_study.add_trial(trial)

new_study.add_trials(complete_trials.trials)

reduced_complete = optuna.create_study()

for i, trial in enumerate(complete_trials.trials, start=0):
    if i not in [1,10]:
        reduced_complete.add_trial(trial)


#fig = optuna.visualization.plot_contour(study, params=["batch_size", "kernel_size","init_features","enc_depth", "dec_depth"])

fig = optuna.visualization.plot_intermediate_values(reduced_complete)
fig.show()
