datasets = ["DTU", "PKU","KUL"]
spa = [0,1,2,3,4,5]

run_file = "run_all.slurm"

with open(run_file, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH -o ./log/%j.out\n")
    f.write("#SBATCH -e ./log/%j.err\n")
    f.write("#SBATCH --ntasks-per-node=32\n")
    f.write("#SBATCH --partition=GPUA800\n")
    f.write("#SBATCH -J spa")
    f.write("#SBATCH --gres=gpu:4\n\n")

    for j in spa:
        model = spa[j]

        for i in range(len(datasets)):

                dataset = datasets[i]

                f.write("python mkdir.py --dataset {} --model RSC &\n".format(dataset, model))
                f.write("python main.py --dataset {} --spa {} --sbfold 0 &\n".format(dataset, model))
                f.write("python main.py --dataset {} --spa {} --sbfold 1 &\n".format(dataset, model))
                f.write("python main.py --dataset {} --spa {} --sbfold 2 &\n".format(dataset, model))
                f.write("python main.py --dataset {} --spa {} --sbfold 3 &\n".format(dataset, model))
                f.write("wait\n")
                f.write("python get_final_res.py --dataset {} --model RSC --spa {}\n".format(dataset, model))
                f.write("\n\n")

