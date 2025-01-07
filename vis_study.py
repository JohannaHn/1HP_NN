import optuna
import matplotlib.pyplot as plt
from optuna.trial import TrialState
import scipy
import numpy as np
from networks.convLSTM import Seq2Seq
import statistics as stat
import os
import logging
from collections import defaultdict

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
    for study in studies:
        for trial in study.trials:
            
            if trial.value is None or len(trial.intermediate_values) != 80:
                continue
            print(len(trial.intermediate_values))
            value = get_best_inter_value(trial)
            if value is not None and value < 0.0006:
                try:
                    num_layers_org.append(trial.params["num_layers"])
                    extend_org.append(trial.params["extend"])
                    prev_boxes_org.append(trial.params["prev_boxes"])
                    values_org.append(value)
                except KeyError:
                    print(f"Key Error in study {study.name}")

    colors = ['#c2e699', '#78c679', '#31a354', '#006837']

    fig, ax = plt.subplots(2, 2, figsize=(10,6))
    list1 = list(zip(extend_org, num_layers_org, values_org, prev_boxes_org))

    # num layers = 1
    extend_2 = list([item[0] for item in list1 if item[1] == 1 and item[3] == 3])
    loss_2 = list([item[2] for item in list1 if item[1] == 1 and item[3] == 3])
    paired = sorted(zip(extend_2, loss_2))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    # x,y = zip(*paired)
    ax[0][0].plot(x,y, color=colors[0], label="1 Layer")
    ax[0][1].scatter(x,y, color=colors[0], s=15, marker="x", label="1 LSTM layer")
    a, b, c = np.polyfit(x, y, deg=2)
    x_lin = np.linspace(1,6,50)
    # ax[0][1].plot(x_lin, a* x_lin**2 + b * x_lin + c)

    # num layers = 2
    extend_2 = list([item[0] for item in list1 if item[1] == 2 and item[3] == 3])
    loss_2 = list([item[2] for item in list1 if item[1] == 2 and item[3] == 3])
    paired = sorted(zip(extend_2, loss_2))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    # x,y =zip(*paired)
    ax[0][0].plot(x,y, color=colors[1], label="2 LSTM layers")
    ax[0][1].scatter(x,y, color=colors[1], s=15, marker='x', label="2 LSTM layers")
    a, b, c = np.polyfit(x, y, deg=2)
    # ax[0][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    # num layers = 3
    extend = list([item[0] for item in list1 if item[1] == 3 and item[3] == 3])
    loss = list([item[2] for item in list1 if item[1] == 3 and item[3] == 3])
    paired = sorted(zip(extend, loss))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    # x,y = zip(*paired)
    ax[0][0].plot(x,y, color=colors[2], label="3 LSTM layers")
    ax[0][1].scatter(x,y, s=15, color=colors[2], marker='x', label="3 LSTM layers")
    a,b,c = np.polyfit(x, y, deg=2)
    # ax[0][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    # num layers = 4
    extend = list([item[0] for item in list1 if item[1] == 4 and item[3] == 3])
    loss = list([item[2] for item in list1 if item[1] == 4 and item[3] == 3])
    paired = sorted(zip(extend, loss))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    # x,y = zip(*paired)
    ax[0][0].plot(x,y, color=colors[3], label="4 LSTM layers")
    ax[0][1].scatter(x,y,s=15, color=colors[3], marker="x", label="4 LSTM layer")
    a,b,c = np.polyfit(x, y, deg=2)
    # ax[0][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    ax[0][0].set_xlabel("Extend Boxes")
    ax[0][0].set_ylabel("Loss")
    ax[0][0].legend(loc='upper right')



    list1 = list(zip(extend_org, prev_boxes_org, values_org, num_layers_org))
    # prev boxes = 1
    extend_2 = list([item[0] for item in list1 if item[1] == 1 and item[3]==3])
    loss_2 = list([item[2] for item in list1 if item[1] == 1 and item[3]==3])
    paired = sorted(zip(extend_2, loss_2))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    # x,y = zip(*paired)
    ax[1][0].plot(x,y, color=colors[0], label="1 Prev Boxes")
    ax[1][1].scatter(x,y,s=15, marker='x', color=colors[0], label="1 Prev Boxes")
    a,b,c = np.polyfit(x, y, deg=2)
    # ax[1][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    # prev boxes = 2
    extend_2 = list([item[0] for item in list1 if item[1] == 2 and item[3]==3])
    loss_2 = list([item[2] for item in list1 if item[1] == 2 and item[3]==3])
    paired = sorted(zip(extend_2, loss_2))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    # x,y = zip(*paired)
    ax[1][0].plot(x,y, color=colors[1], label="2 Prev Boxes")
    ax[1][1].scatter(x,y,s=15, marker='x', color=colors[1], label="2 Prev Boxes")
    a,b,c = np.polyfit(x, y, deg=2)
    # ax[1][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)
    ax[1][1].set_ylim(0,0.0003)
    ax[0][1].set_ylim(0.00004,0.000125)

    # prev boxes = 3
    extend = list([item[0] for item in list1 if item[1] == 3 and item[3]==3])
    loss = list([item[2] for item in list1 if item[1] == 3 and item[3]==3])
    paired = sorted(zip(extend, loss))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    # x,y = zip(*paired)
    ax[1][0].plot(x,y, color=colors[2], label="3 Prev Boxes")
    ax[1][1].scatter(x,y, s=15, marker='x', color=colors[2], label="3 Prev Boxes")
    a,b,c = np.polyfit(x, y, deg=2)
    # ax[1][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)

    # prev boxes = 4
    extend = list([item[0] for item in list1 if item[1] == 4 and item[3]==3])
    loss = list([item[2] for item in list1 if item[1] == 4 and item[3]==3])
    paired = sorted(zip(extend, loss))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    # x,y = zip(*paired)
    ax[1][0].plot(x,y, color=colors[3], label="4 Prev Boxes")
    ax[1][1].scatter(x,y,s=15, marker='x', color=colors[3],label="4 Prev Boxes")
    a,b,c = np.polyfit(x, y, deg=2)
    # ax[1][1].plot(x_lin, a*x_lin**2 + b * x_lin + c)


    ax[1][0].set_xlabel("Extend Boxes")
    ax[1][0].set_ylabel("Validation loss")
    
    ax[1][0].legend(loc='upper right')
    fig.savefig("/home/hofmanja/1HP_NN/plot_study.png")


def plot_analysis_scatter():
    num_layers_org = []
    extend_org = []
    prev_boxes_org = []
    values_org = []
    
    # Extract data from studies
    for study in studies:
        for trial in study.trials:
            if trial.value is None or len(trial.intermediate_values) != 80:
                continue
            value = get_best_inter_value(trial)
            if value is not None and value < 0.0006:
                try:
                    num_layers_org.append(trial.params["num_layers"])
                    extend_org.append(trial.params["extend"])
                    prev_boxes_org.append(trial.params["prev_boxes"])
                    values_org.append(value)
                except KeyError:
                    print(f"Key Error in study {study.name}")

    colors = ['#c2e699', '#78c679', '#31a354', '#006837']

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    list1 = list(zip(extend_org, num_layers_org, values_org, prev_boxes_org))

    # Plot for num_layers = 1 (ax[0][1])
    extend_2 = [item[0] for item in list1 if item[1] == 1 and item[3] == 3]
    loss_2 = [item[2] for item in list1 if item[1] == 1 and item[3] == 3]
    paired = sorted(zip(extend_2, loss_2))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    ax[0].scatter(x, y, color=colors[0], s=40, marker="x", label="1 Layer")

    # num layers = 2
    extend_2 = list([item[0] for item in list1 if item[1] == 2 and item[3] == 3])
    loss_2 = list([item[2] for item in list1 if item[1] == 2 and item[3] == 3])
    paired = sorted(zip(extend_2, loss_2))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    ax[0].scatter(x, y, color=colors[1], s=40, marker="x", label="2 Layers")
    
    # num layers = 3
    extend = list([item[0] for item in list1 if item[1] == 3 and item[3] == 3])
    loss = list([item[2] for item in list1 if item[1] == 3 and item[3] == 3])
    paired = sorted(zip(extend, loss))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    ax[0].scatter(x, y, color=colors[2], s=40, marker="x", label="3 Layers")

    # num layers = 4
    extend = list([item[0] for item in list1 if item[1] == 4 and item[3] == 3])
    loss = list([item[2] for item in list1 if item[1] == 4 and item[3] == 3])
    paired = sorted(zip(extend, loss))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    ax[0].scatter(x,y,s=40, color=colors[3], marker="x", label="4 Layers")

    ax[0].set_ylim(0.00004, 0.000125)
    ax[0].set_xlabel("Predicted Frames (Dec length)")
    ax[0].set_ylabel("Validation loss")
    ax[0].legend(loc='upper right', bbox_to_anchor=(1,1))

    list1 = list(zip(extend_org, prev_boxes_org, values_org, num_layers_org))
    # Plot for prev_boxes = 1 (ax[1][1])
    extend_2 = [item[0] for item in list1 if item[1] == 1 and item[3] == 3]
    loss_2 = [item[2] for item in list1 if item[1] == 1 and item[3] == 3]
    paired = sorted(zip(extend_2, loss_2))
    print(paired)
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    print(x)
    print(y)
    ax[1].scatter(x, y, s=40, marker='x', color=colors[0], label="Enc length 1")

    # prev boxes = 2
    extend_2 = list([item[0] for item in list1 if item[1] == 2 and item[3]==3])
    loss_2 = list([item[2] for item in list1 if item[1] == 2 and item[3]==3])
    paired = sorted(zip(extend_2, loss_2))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    ax[1].scatter(x, y, s=40, marker='x', color=colors[1], label="Enc length 2")

    # prev boxes = 3
    extend = list([item[0] for item in list1 if item[1] == 3 and item[3]==3])
    loss = list([item[2] for item in list1 if item[1] == 3 and item[3]==3])
    paired = sorted(zip(extend, loss))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    ax[1].scatter(x, y, s=40, marker='x', color=colors[2], label="Enc length 3")

    # prev boxes = 4
    extend = list([item[0] for item in list1 if item[1] == 4 and item[3]==3])
    loss = list([item[2] for item in list1 if item[1] == 4 and item[3]==3])
    paired = sorted(zip(extend, loss))
    loss_dict = defaultdict(list)
    for e, l in paired:
        loss_dict[e].append(l)
    mean_loss_dict = {e: sum(losses) / len(losses) for e, losses in loss_dict.items()}
    x, y = zip(*mean_loss_dict.items())
    ax[1].scatter(x,y,s=40, marker='x', color=colors[3],label="Enc length 3")
    
    ax[1].set_ylim(0, 0.0004)
    ax[1].set_xlabel("Predicted Frames (Dec length)")
    ax[1].set_ylabel("Validation loss")
    ax[0].set_ylabel("Validation loss")
    ax[1].legend(loc='upper right', bbox_to_anchor=(1,1))
    ax[0].set_facecolor('#B0B0B0')
    ax[1].set_facecolor('#B0B0B0')
    ax[0].set_title("Influence of Layers")
    ax[1].set_title("Influence of Encoder length")

    for axis in ax.flat:
        axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    fig.savefig("/home/hofmanja/1HP_NN/plot_analysis_scatter.png", dpi=500)



def get_best_inter_value(trial):
    best = trial.value
    for i in range(len(trial.intermediate_values)):
        if trial.intermediate_values[i] < best:
            best = trial.intermediate_values[i]

    return best


def plot_slice(study):
    best_value = 1
    best_params = [1 for i in range(8)]
    params = ["batch_size", "lr", "enc_depth", "dec_depth", "init_features", "kernel_size", "num_layers", "prev_boxes"]
    fig, ax = plt.subplots(2,4,sharey='row')
    for trial in study.trials:
        value = get_best_inter_value(trial)
        if value < best_value:
            best_value = value
            for i in range(8):
                best_params[i] = trial.params[params[i]]

        ax[0][0].scatter(trial.params["batch_size"], value, marker='x', s=15, color="blue", alpha=0.5)
        ax[0][0].set_xlabel("Batch size")
        ax[0][0].set_ylabel("Validation loss")
        ax[0][0].set_xticks([16,32])
        ax[0][0].set_xlim(8,40)
        if trial.params["lr"] < 0.01:
            ax[0][1].scatter(trial.params["lr"], value, marker='x', s=15, color="blue", alpha=0.5)
        ax[0][1].set_xlabel("Learning rate")
        
        ax[0][1].set_xticks([0.0002, 0.003])
        ax[0][1].set_xticklabels([r'$2 \cdot 10^{-4}$', r'$3 \cdot 10^{-3}$'])
        ax[0][2].scatter(trial.params["enc_depth"], value , marker='x', s=15, color="blue", alpha=0.5)
        ax[0][2].set_xlabel("Enc. conv. layers")
        ax[0][2].set_xticks([4,5,6,7])
        ax[0][2].set_xlim(3.5,7.5)
        ax[0][3].scatter(trial.params["dec_depth"], value, marker='x', s=15, color="blue", alpha=0.5)
        ax[0][3].set_xlabel("Dec. conv. layers")
        ax[0][3].set_xticks([4,5,6])
        ax[0][3].set_xlim(3.5, 6.5)
        ax[1][0].scatter(trial.params["init_features"], value,marker='x', s=15, color="blue", alpha=0.5)
        ax[1][0].set_xlabel("Initial features")
        ax[1][0].set_ylabel("Validation loss")
        ax[1][0].set_xticks([16,32,64])
        ax[1][0].set_xlim(8,72)
        ax[1][1].scatter(trial.params["kernel_size"], value, marker='x', s=15, color="blue", alpha=0.5)
        ax[1][1].set_xlabel("Kernel size")
        ax[1][1].set_xticks([3,5,7,9])
        ax[1][1].set_xlim(2.5, 9.5)
        ax[1][2].scatter(trial.params["num_layers"], value, marker='x', s=15, color="blue", alpha=0.5)
        ax[1][2].set_xlabel("Layers")
        ax[1][2].set_xticks([2,3,4])
        ax[1][2].set_xlim(1.5, 4.5)
        ax[1][3].scatter(trial.params["prev_boxes"], value, marker='x', s=15, color="blue", alpha=0.5)
        ax[1][3].set_xlabel("Encoder length")
        ax[1][3].set_xticks([1,2,3])
        ax[1][3].set_xlim(0.5, 3.5)

    
    for axis in ax.flat:
        #axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x * 1e4)}'))
        axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    params = [16, 0.0004, 4, 4, 64, 5, 2, 3]

    for i, axis in enumerate(ax.flat):
        axis.scatter(best_params[i], best_value, marker='x', s=15, color="red")

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    fig.savefig("/home/hofmanja/1HP_NN/plot_hyperparams.png", dpi=500)


study = optuna.load_study(study_name="study", storage="sqlite:///path_to_study/study.db")
studies = [study]
 
plot_analysis_scatter()

plot_analysis()

fig = optuna.visualization.plot_timeline(study)

