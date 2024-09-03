import argparse
from datetime import datetime
import os
from pathlib import Path
import torch
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from torch.nn import MSELoss, L1Loss
import traceback
import numpy as np

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import data_stuff.utils as utils
import utils.utils_args as ut
from main import init_data
from data_stuff.dataset import SimulationDataset, DatasetExtend1, DatasetExtend2, DatasetExtendConvLSTM, get_splits
from data_stuff.utils import SettingsTraining
from networks.convLSTM import Seq2Seq
from processing.solver import Solver
from preprocessing.prepare import prepare_data_and_paths
from postprocessing.visualization import plot_avg_error_cellwise, visualizations, infer_all_and_summed_pic, visualizations_convLSTM
from postprocessing.measurements import measure_loss, save_all_measurements, measure_losses_paper24

def objective(trial):

    utils.save_yaml(settings, settings.destination, "command_line_arguments.yaml")
    

    extend = 2

    prev_boxes = trial.suggest_int("prev_boxes", 1, 3)
    settings.prev_boxes = prev_boxes
    
    enc_depth = trial.suggest_int("enc_depth", 4, 7)
    dec_depth = trial.suggest_int("dec_depth", 4, 7)

    kernel_size = trial.suggest_int("kernel_size", 3, 9, step=2)
    enc_kernel_sizes = [kernel_size for _ in range(enc_depth)]
    dec_kernel_sizes = [kernel_size for _ in range(dec_depth)]
    
    init_features = trial.suggest_categorical("init_features", [16, 32, 64])

    enc_conv_features = np.array([init_features, *[64 for _ in range(enc_depth-1)]])
    dec_conv_features = [64 for _ in range(dec_depth)]

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Get the dataset.
    _, dataloaders = init_data(settings, batch_size)

    # Generate the model.
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"]) #practical reasoning: dont allow negative values (Leaky ReLU)

    num_layers = trial.suggest_categorical("num_layers", [2,3,4])

    # trial.suggest_categorical("prev_boxes", [1,2,3])

    model = Seq2Seq(num_channels=3, frame_size=(64,64), prev_boxes =prev_boxes, 
                            extend=extend, 
                            num_layers=num_layers,
                            enc_conv_features=enc_conv_features,
                            enc_kernel_sizes=enc_kernel_sizes,
                            dec_conv_features=dec_conv_features,
                            dec_kernel_sizes=dec_kernel_sizes,
                            activation=activation).float()
    model.to(settings.device)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"]) #, "RMSprop"]) #optimized, "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=MSELoss(), learning_rate=lr)
    try:
        training_time = datetime.now()
        solver.load_lr_schedule(settings.destination / "learning_rate_history.csv")
        loss = solver.train(trial, settings)
        training_time = datetime.now() - training_time
        solver.save_lr_schedule(settings.destination / "learning_rate_history.csv")
        model.save(settings.destination, model_name = f"model_trial_{trial.number}.pt")
        solver.save_metrics(settings.destination, model.num_of_params(), settings.epochs, training_time, settings.device)

        dataloader = dataloaders["val"]
        #visualizations_convLSTM(model, dataloaders['test'], settings.device, prev_boxes=settings.prev_boxes, extend=settings.extend, plot_path=settings.destination, dp_to_visu=1, pic_format='png')
    except Exception as e:
        loss = 1
        if isinstance(e, torch.cuda.OutOfMemoryError):
            print(trial.params)
            raise optuna.exceptions.TrialPruned("Training failed due to CUDA out of memory.")
        else:
            traceback.print_exc()
            raise optuna.exceptions.TrialPruned(f"Training failed due to {e}")
            
    try:
        metrics = measure_losses_paper24(model, dataloaders, args)
        ut.save_yaml(metrics, f'{settings.destination}/metrics_trial_{trial.number}.yaml')
    except Exception as e:
        print("Could not measure losses")
        pass

    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="extend_plumes/ep_medium_1000dp_only_vary_dist", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_prep", type=str, default="extend_plumes/ep_medium_1000dp_only_vary_dist inputs_ks")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune"], default="train")
    parser.add_argument("--model", type=str, default="default") # required for testing or finetuning
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--inputs", type=str, default="ks") #choices=["gki", "gksi", "pksi", "gks", "gksi100", "ogksi1000", "gksi1000", "pksi100", "pksi1000", "ogksi1000_finetune", "gki100", "t", "gkiab", "gksiab", "gkt"]
    parser.add_argument("--problem", type=str, choices=["2stages", "allin1", "extend1", "extend2",], default="extend_plumes")
    parser.add_argument("--prev_boxes", type=int, default=1)
    parser.add_argument("--extend", type=int, default=2)
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--loss", type=str, choices=['mse', 'l1'], default='mse')
    parser.add_argument("--enc_conv_features", default=[16, 32, 64, 64, 64])
    parser.add_argument("--dec_conv_features", default=[64, 64, 64])
    parser.add_argument("--enc_kernel_sizes", default = [7, 5, 5, 5, 5])
    parser.add_argument("--dec_kernel_sizes", default=[5, 5, 7])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--skip_per_dir", type=int, default=64)
    args = parser.parse_args()
    settings = SettingsTraining(**vars(args))

    settings = prepare_data_and_paths(settings)

    study_name = "second"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
   # optuna.delete_study(study_name=study_name, storage=storage_name)
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name, load_if_exists=True)

    study.optimize(objective, n_trials=25, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    print("Done")