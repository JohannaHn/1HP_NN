import optuna
import matplotlib.pyplot as plt
from optuna.trial import TrialState
import scipy
import numpy as np
from networks.convLSTM import Seq2Seq
import statistics as stat
import os
import logging

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

def get_smallest_within(values, limit):
    for value in sorted(values):
        if value >= limit:
            return value

def get_largest_within(values, limit):
    for value in reversed(sorted(values)):
        if value <= limit:
            return value

def plot_box_plot(studies):
    fix, ax = plt.subplots(1,1,figsize=(8,8))
    map = {16: 2, 8: 1, 32: 3}
    mapped_batches_tot = []
    values_tot = []
    for i, study in enumerate(studies):
        batch_sizes = []
        values = []

        for trial in study.trials:
            try:
                batch_size = trial.params["batch_size"]
            except KeyError:
                try:
                    batch_size=trial.params["batch_size_fixed_at"]
                except KeyError:
                    continue
            if trial.value is not None and trial.value < 0.004:
                batch_sizes.append(batch_size)
                values.append(trial.value)
        

        mapped_batches = [map[key] for key in batch_sizes]
        jitter = (np.random.rand(len(mapped_batches))-0.5 ) / 3
        ax.scatter(mapped_batches+jitter, values, s=8, alpha=0.5, label=study.study_name)
        mapped_batches_tot.append(mapped_batches)
        values_tot.append(values)
        if i == 1:
            print(mapped_batches_tot)
            print(*mapped_batches_tot)

    mapped_batches_tot = [item for sublist in mapped_batches_tot for item in sublist]
    values_tot = [item for sublist in values_tot for item in sublist]
    for i in [1,2,3]:
        filtered_values = [v for k,v in zip(mapped_batches_tot, values_tot) if k==i]
        if len(filtered_values) > 0:
            median = stat.median(filtered_values)
            q1 = np.percentile(filtered_values, 25)
            q3 = np.percentile(filtered_values, 75)
            ax.plot([i-0.1, i+0.1], [median, median], c='black')
            ax.plot([i-0.1, i-0.1, i-0.1, i+0.1, i+0.1, i+0.1, i-0.1], [q1, median, q3, q3, median, q1, q1], c='black')
            lower_fence = get_smallest_within(filtered_values, q1 - 1.5*(q3-q1))
            upper_fence = get_largest_within(filtered_values, q3 + 1.5*(q3-q1))
            ax.plot([i-0.1, i+0.1], [lower_fence, lower_fence], c='black')
            ax.plot([i-0.1, i+0.1], [upper_fence, upper_fence], c='black')
            ax.plot([i, i], [q3, upper_fence], c='black')
            ax.plot([i, i], [q1, lower_fence], c='black')


    ax.legend(loc='upper right')
    ax.set_xticks([1,2,3], ['8', '16', '32'])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Loss")
    plt.savefig("batch_sizes_scatter.png")

def remove_outliers(study):
    trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    
    losses = np.array([trial.value for trial in trials])
    
    # Compute the first quartile (Q1) and third quartile (Q3)
    q1 = np.percentile(losses, 25)
    q3 = np.percentile(losses, 75)
    
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    
    # Define the range for non-outliers
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out trials whose losses fall outside of the non-outlier range

    new_study = optuna.create_study()
    for trial in trials:
        if trial.value <= upper_bound:
            new_study.add_trial(trial)

    return new_study

def plot_analysis():

    num_layers_org = []
    extend_org = []
    prev_boxes_org = []
    values_org = []
    for trial in layers_prev_extend.trials:
        if trial.value is not None and trial.value < 1:
            try:
                num_layers_org.append(trial.params["num_layers"])
                extend_org.append(trial.params["extend"])
                prev_boxes_org.append(trial.params["prev_boxes"])
                values_org.append(trial.value)
            except KeyError:
                continue


    fig, ax = plt.subplots(2, 2, figsize=(10,6))
    list1 = list(zip(extend_org, num_layers_org, values_org, prev_boxes_org))

    # num layers = 1
    extend_2 = list([item[0] for item in list1 if item[1] == 1 and item[3] == 3])
    loss_2 = list([item[2] for item in list1 if item[1] == 1 and item[3] == 3])
    paired = sorted(zip(extend_2, loss_2))
    x, y = zip(*paired)
    ax[0][0].plot(x,y, label="1 LSTM layer")
    ax[0][1].scatter(x,y, s=5, alpha=0.5, label="1 LSTM layer")
    a, b, c = np.polyfit(x, y, deg=2)
    x_lin = np.linspace(1,6,50)
    ax[0][1].plot(x_lin, a* x_lin**2 + b * x_lin + c)

    # num layers = 2
    extend_2 = list([item[0] for item in list1 if item[1] == 2 and item[3] == 3])
    loss_2 = list([item[2] for item in list1 if item[1] == 2 and item[3] == 3])
    paired = sorted(zip(extend_2, loss_2))
    x, y = zip(*paired)
    ax[0][0].plot(x,y, label="2 LSTM layers")
    ax[0][1].scatter(x,y, s=5, alpha=0.5, label="2 LSTM layers")
    a, b, c = np.polyfit(x, y, deg=2)
    ax[0][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    # num layers = 3
    extend = list([item[0] for item in list1 if item[1] == 3 and item[3] == 3])
    loss = list([item[2] for item in list1 if item[1] == 3 and item[3] == 3])
    paired = sorted(zip(extend, loss))
    x, y = zip(*paired)
    ax[0][0].plot(x,y, label="3 LSTM layers")
    ax[0][1].scatter(x,y, s=15, alpha=0.5, label="3 LSTM layers")
    a,b,c = np.polyfit(x, y, deg=2)
    ax[0][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    # num layers = 4
    extend = list([item[0] for item in list1 if item[1] == 4 and item[3] == 3])
    loss = list([item[2] for item in list1 if item[1] == 4 and item[3] == 3])
    paired = sorted(zip(extend, loss))
    x, y = zip(*paired)
    ax[0][0].plot(x,y, label="4 LSTM layers")
    ax[0][1].scatter(x,y,s=5, alpha=0.5, label="4 LSTM layer")
    a,b,c = np.polyfit(x, y, deg=2)
    ax[0][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    ax[0][0].set_xlabel("Extend Boxes")
    ax[0][0].set_ylabel("Loss")
    ax[0][0].legend(loc='upper right')



    list1 = list(zip(extend_org, prev_boxes_org, values_org, num_layers_org))
    # prev boxes = 1
    extend_2 = list([item[0] for item in list1 if item[1] == 1 and item[3]==3])
    loss_2 = list([item[2] for item in list1 if item[1] == 1 and item[3]==3])
    paired = sorted(zip(extend_2, loss_2))
    x, y = zip(*paired)
    ax[1][0].plot(x,y, label="1 Prev Boxes")
    ax[1][1].scatter(x,y,s=5, alpha=0.5, label="1 Prev Boxes")
    a,b,c = np.polyfit(x, y, deg=2)
    ax[1][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    # prev boxes = 2
    extend_2 = list([item[0] for item in list1 if item[1] == 2 and item[3]==3])
    loss_2 = list([item[2] for item in list1 if item[1] == 2 and item[3]==3])
    paired = sorted(zip(extend_2, loss_2))
    x, y = zip(*paired)
    ax[1][0].plot(x,y, label="2 Prev Boxes")
    ax[1][1].scatter(x,y,s=5, alpha=0.5, label="2 Prev Boxes")
    a,b,c = np.polyfit(x, y, deg=2)
    ax[1][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    # prev boxes = 3
    extend = list([item[0] for item in list1 if item[1] == 3 and item[3]==3])
    loss = list([item[2] for item in list1 if item[1] == 3 and item[3]==3])
    paired = sorted(zip(extend, loss))
    x, y = zip(*paired)
    ax[1][0].plot(x,y, label="3 Prev Boxes")
    ax[1][1].scatter(x,y,s=15, alpha=0.5, label="3 Prev Boxes")
    a,b,c = np.polyfit(x, y, deg=2)
    ax[1][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)
    # prev boxes = 4
    extend = list([item[0] for item in list1 if item[1] == 4 and item[3]==3])
    loss = list([item[2] for item in list1 if item[1] == 4 and item[3]==3])
    paired = sorted(zip(extend, loss))
    x, y = zip(*paired)
    ax[1][0].plot(x,y, label="4 Prev Boxes")
    ax[1][1].scatter(x,y,s=5, alpha=0.5, label="4 Prev Boxes")
    a,b,c = np.polyfit(x, y, deg=2)
    ax[1][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    ax[1][0].set_xlabel("Extend Boxes")
    ax[1][0].set_ylabel("Loss")
    ax[1][0].legend(loc='upper right')
    plt.savefig("plot_study.png")



logging.getLogger("optuna").setLevel(logging.WARNING)
databases = "/home/hofmanja/1HP_NN/databases"
database_names = os.listdir(databases)

plotted = ["with_metrics","second","layers_prev_extend","layers_prev_extend_without_pruned", "kernel_size_evolution", "small_batch"]
#plotted = []

# for db in database_names:
#     db_path = os.path.join(databases, db)  # Use os.path.join to construct the full path
#     name_without_extension = os.path.splitext(db)[0]  # Get the name without extension
#     storage_url = f"sqlite:///{db_path}"  # Correctly format the storage URL

#     if name_without_extension not in plotted:
#         try:
#             study = optuna.load_study(study_name=name_without_extension, storage=storage_url)
#             study = trials_below(study, 1)
#             fig = optuna.visualization.plot_slice(study)
#             fig.show()
#             fig = optuna.visualization.plot_intermediate_values(study)
#             fig.show()
#             print(f"{name_without_extension} plotted.")
#             plotted.append(name_without_extension)
#         except Exception as e:
#             print(f"Error loading {name_without_extension}:", e)

# db_path = "/home/hofmanja/1HP_NN/databases/second.db"  # Use os.path.join to construct the full path
# name_without_extension = "second"  # Get the name without extension
# storage_url = f"sqlite:///{db_path}" 
# study = optuna.load_study(study_name=name_without_extension, storage=storage_url)
# print(study.best_params)
        
studies = []
for db in database_names:
    if db in ["layers_prev_extend_without_pruned.db"]:
        db_path = os.path.join(databases, db)  # Use os.path.join to construct the full path
        name_without_extension = os.path.splitext(db)[0]  # Get the name without extension
        storage_url = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=name_without_extension, storage=storage_url)
        studies.append(study)

# plot_box_plot(studies)

#remove outliers
layers_prev_extend = optuna.load_study(study_name="layers_prev_extend_without_pruned", storage="sqlite:////home/hofmanja/1HP_NN/databases/layers_prev_extend_without_pruned.db")
layers_prev_extend2 = optuna.load_study(study_name="layers_prev_extend_without_pruned", storage="sqlite:////home/hofmanja/1HP_NN/layers_prev_extend_without_pruned.db")

for trial in layers_prev_extend2.trials:
    if trial.params["extend"] == 2 and trial.params["num_layers"] == 2 and trial.params["prev_boxes"] == 3:
        print(trial.value)
layers_prev_extend.add_trials(layers_prev_extend2.trials)    
layers_prev_extend = remove_outliers(layers_prev_extend)
plot_analysis()

storage = "sqlite:////home/hofmanja/1HP_NN/databases/second.db"

# Get all study summaries
study_summaries = optuna.study.get_all_study_summaries(storage)

# Print the study summaries
for summary in study_summaries:
    print(summary)


# fig = optuna.visualization.plot_timeline(layers_prev_extend)
# fig = optuna.visualization.plot_timeline(layers_prev_extend)
# fig.show()
# fig = optuna.visualization.plot_slice(layers_prev_extend)
# fig.show()