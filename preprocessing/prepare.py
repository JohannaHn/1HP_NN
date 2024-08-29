import shutil
import torch
from torch.utils.data import DataLoader, random_split

from data_stuff.utils import SettingsTraining
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from preprocessing.prepare_paths import Paths1HP,  set_paths_1hpnn
from data_stuff.dataset import DatasetExtendConvLSTM, get_splits

def init_data(args:dict, seed=1):
    dataset = DatasetExtendConvLSTM(args["dataset_prep"], prev_steps=args["prev_boxes"], extend=args["extend"], skip_per_dir=32)

        # dataset = DatasetEncoder(args["data_prep"], box_size=args["len_box"], skip_per_dir=args["skip_per_dir"])
    args["inputs"] += "T"
            # dataset_val = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], idx=1)

    split_ratios = [0.7, 0.2, 0.1]
    generator = torch.Generator().manual_seed(seed)
    datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)

    dataloaders = {}
    batchsize = 20
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=batchsize, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=batchsize, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=batchsize, shuffle=True, num_workers=0)

    return dataloaders

def prepare_data_and_paths(settings:SettingsTraining):
    if settings.case_2hp:
        assert False, "not implemented on this branch"
    else:
        paths: Paths1HP
        paths, destination_dir = set_paths_1hpnn(settings.dataset_raw, settings.inputs, settings.dataset_prep, problem=settings.problem) 
        settings.dataset_prep = paths.dataset_1st_prep_path

        settings.make_destination_path(destination_dir)
        settings.save_notes()
        settings.make_model_path(destination_dir)

        # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
        if not settings.dataset_prep.exists(): # or settings.case == "test": # if test, always want to prepare because the normalization parameters have to match
            prepare_dataset_for_1st_stage(paths, settings)
        print(f"Dataset prepared ({paths.dataset_1st_prep_path})")


    if settings.case == "train":
        shutil.copyfile(paths.dataset_1st_prep_path / "info.yaml", settings.destination / "info.yaml")
    settings.save()
    return settings