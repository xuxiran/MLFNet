datasets = ["DTU", "PKU","KUL"]
tems_all = [0,125,250,375,500,625,750,875]
teme_all = [1000,1000,1000,1000,1000,1000,1000,1000]

run_file = "run_all2.slurm"
with open(run_file, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH -o ./log/%j.out\n")
    f.write("#SBATCH -e ./log/%j.err\n")
    f.write("#SBATCH --ntasks-per-node=32\n")
    f.write("#SBATCH --partition=GPUA800\n")
    f.write("#SBATCH -J temp2\n")
    f.write("#SBATCH --gres=gpu:4\n\n")

    for i in range(len(tems_all)):
        tems = tems_all[i]
        teme = teme_all[i]


        for dataset in datasets:
            f.write("python mkdir.py --dataset {} --model RSC &\n".format(dataset))
            f.write("python main.py --dataset {} --sbfold 0 --tems {} --teme {} &\n".format(dataset, tems, teme))
            f.write("python main.py --dataset {} --sbfold 1 --tems {} --teme {} &\n".format(dataset, tems, teme))
            f.write("python main.py --dataset {} --sbfold 2 --tems {} --teme {} &\n".format(dataset, tems, teme))
            f.write("python main.py --dataset {} --sbfold 3 --tems {} --teme {} &\n".format(dataset, tems, teme))
            f.write("wait\n")
            f.write("python get_final_res.py --dataset {} --model RSC --tems {} --teme {}\n".format(dataset, tems, teme))
            f.write("\n\n")

