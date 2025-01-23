
---

# MLFNet

This repository contains the complete code for MLFNet. The structure is organized into six main folders:

### Folder Structure

- **MLFNet_05s**: Contains models for decoding accuracy with a 0.5s decision window.
- **MLFNet_1s**: Contains models for decoding accuracy with a 1s decision window.
- **MLFNet_2s**: Contains models for decoding accuracy with a 2s decision window.

#### Inside Each Folder

- **BASEmodels**: Stores six reproduced models:
  - CNN
  - STANet
  - XANet
  - DenseNet
  - DBPNet
  - DARNet

- **OURmodels**: Stores our proposed MLFNet model and models used for ablation experiments:
  - **RSC**: The complete MLFNet trained with the RSC strategy.
  - **Base**: Ablation of the RSC training strategy.
  - **Nocat**: Ablation of multi-level fusion.
  - **Nofre**: Ablation of frequency dimension input.
  - **Notem**: Ablation of temporal dimension input.

### Main Files

- **main.py**: The main program to run the complete results.
- **config.py**: Stores various hyperparameters.
- **AADdataset.py**: Constructs the training, validation, and test datasets.
- **get_model.py**: Loads the models used in this study.
- **mkdir.py**: Creates the necessary directories.
- **get_final_res.py**: Aggregates the results from each GPU.

### Analysis Code

- **MLFNet_analysis_fre.py**: Code for spectral analysis as described in Section 5.1.
- **MLFNet_analysis_tem.py**: Code for temporal analysis as described in Section 5.1.
- **MLFNet_analysis_spa.py**: Code for spatial analysis as described in Section 5.1.

Each `.py` file corresponds to the analyses mentioned above.

### Additional Information

- We used four A800 GPUs to train the models, running one fold on each GPU to accelerate the process.
- Please use the `requirements.txt` file in the current directory to create a virtual environment.

### Example Slurm File

A complete task slurm file is provided below. To run more results, you can adjust `gen_run.py` or `gen_run_all.py` to generate slurm files directly.

---

This structured README provides a clear overview of the repository, making it easier for users to navigate and understand the contents and purpose of each file and folder.

#!/bin/bash
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err
#SBATCH --ntasks-per-node=32
#SBATCH --partition=GPUA800
#SBATCH -J alldataset
#SBATCH --gres=gpu:4

python mkdir.py --dataset DTU --model RSC &
python main.py --dataset DTU --model RSC --sbfold 0 &
python main.py --dataset DTU --model RSC --sbfold 1 &
python main.py --dataset DTU --model RSC --sbfold 2 &
python main.py --dataset DTU --model RSC --sbfold 3 &
wait
python get_final_res.py --dataset DTU --model RSC


python mkdir.py --dataset PKU --model RSC &
python main.py --dataset PKU --model RSC --sbfold 0 &
python main.py --dataset PKU --model RSC --sbfold 1 &
python main.py --dataset PKU --model RSC --sbfold 2 &
python main.py --dataset PKU --model RSC --sbfold 3 &
wait
python get_final_res.py --dataset PKU --model RSC


python mkdir.py --dataset KUL --model RSC &
python main.py --dataset KUL --model RSC --sbfold 0 &
python main.py --dataset KUL --model RSC --sbfold 1 &
python main.py --dataset KUL --model RSC --sbfold 2 &
python main.py --dataset KUL --model RSC --sbfold 3 &
wait
python get_final_res.py --dataset KUL --model RSC
