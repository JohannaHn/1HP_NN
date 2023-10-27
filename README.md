# Python Requirements
- pyyaml
- torch
- h5py
- tensorboard
- tqdm
- pytest

# Begin working
- clone the repository
- install the requirements
- download the raw / prepared data, (optional models and data sets for 2nd stage) and set the paths in paths.yaml (see later)

## Training a 1st stage model (1HP-NN):
- run main.py

    ```
    python main.py --dataset NAME_OF_DATASET
    ```
## Infer a 1st stage model:
- run main.py:

    ```
    python main.py --dataset NAME_OF_DATASET --case test --model PATH_TO_MODEL (after "runs/")
    
    optional arguments:
    --inputs: make sure, they are the same as in the model (default `gksi`)
    --visualize: visualize the results (default `False`)
    ```
## Training a 2nd stage model (2HP-NN):
- for running a 2HP-NN you need the prepared 2HP-dataset in datasets_prepared_dir_2hp
- for preparing 2HP-NN: expects that 1HP-NN exists and trained on; for 2HP-NN (including preparation) run main.py with the following arguments:

    ```
    python main.py --dataset NAME_OF_DATASET --case_2hp True --model PATH_TO_1HPNN_MODEL (after "runs/") --inputs INPUTS (rather preparation case from 1HP-NN)
    ```
    more information on required arguments:
    --inputs: make sure, they are the same as in the model (default `gksi`) + the number of datapoints (e.g. gksi_1000dp)

    optional arguments:
    --visualize: visualize the results (default `False`)
    --case: `test`, `train` or `finetune` (default `train`)

#TODO

## Infer a 2nd stage model:

## Exemplary paths.yaml file:

    ```
    default_raw_dir: /scratch/sgs/pelzerja/datasets # where the raw 1st stage data is stored
    datasets_prepared_dir: /home/pelzerja/pelzerja/test_nn/datasets_prepared/1HP_NN # where the prepared 1st stage data is stored
    datasets_raw_domain_dir: /scratch/sgs/pelzerja/datasets/2hps_demonstrator_copy_of_local
    datasets_prepared_domain_dir: /home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_domain
    prepared_1hp_best_models_and_data_dir: /home/pelzerja/pelzerja/test_nn/1HP_NN_preparation_BEST_models_and_data
    models_2hp_dir: /home/pelzerja/pelzerja/test_nn/1HP_NN/runs
    datasets_prepared_dir_2hp: /home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_NN
    ```

# GPU support
- if you want to use the GPU, you need to install pflotran with cuda support
- check nvidia-smi for the available gpus and cuda version
- `export CUDA_VISIBLE_DEVICES=<gpu_id>` (e.g. 0)
- if the gpu is not found after suspension, try

    `sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm`

    if it does not help, you have to reboot

# important commits
- directly after paper submission (Oct. '23): cdc41426184756b9b1870e5c0f52d399bee0fae0
- after clean up, one month after paper submission (Oct. '23): c8da3da