import optuna
import matplotlib

# Specify the study name and database URL
study_name = "with_metrics"  # Name of the study
storage_url = "sqlite:///{}.db".format(study_name)  # URL of the SQLite database

# List all studies in the database
summaries = optuna.study.get_all_study_summaries(storage=storage_url)
print("Available studies:")
for summary in summaries:
    print(summary.study_name)


study = optuna.load_study(study_name=study_name, storage=f"sqlite:////home/hofmanja/1HP_NN/{study_name}.db")

# Now you can interact with the study object, for example, to see the best trial
print("Best trial:")
for trial in study.trials:
    print(trial.number)
    print(trial.params)

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
