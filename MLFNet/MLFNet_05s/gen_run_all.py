datasets = ["DTU", "PKU","KUL"]
models = ["CNN","STANet","XANet","DenseNet","DARNet","DBPNet","Nofre","Notem","Nocat","Dft","BASE","RSC"]
# models = ["BASE","RSC"]

run_file = "run_all.slurm"
with open(run_file, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("#SBATCH -o ./log/%j.out\n")
    f.write("#SBATCH -e ./log/%j.err\n")
    f.write("#SBATCH --ntasks-per-node=32\n")
    f.write("#SBATCH --partition=GPUA800\n")
    f.write("#SBATCH -J alldataset\n")
    f.write("#SBATCH --gres=gpu:4\n\n")

    for i in range(len(datasets) * len(models)):
        dataset = datasets[i % 3]
        model = models[i // 3]
        f.write("python mkdir.py --dataset {} --model {} &\n".format(dataset, model))
        f.write("python main.py --dataset {} --model {} --sbfold 0 &\n".format(dataset, model))
        f.write("python main.py --dataset {} --model {} --sbfold 1 &\n".format(dataset, model))
        f.write("python main.py --dataset {} --model {} --sbfold 2 &\n".format(dataset, model))
        f.write("python main.py --dataset {} --model {} --sbfold 3 &\n".format(dataset, model))
        f.write("wait\n")
        f.write("python get_final_res.py --dataset {} --model {}\n".format(dataset, model))
        f.write("\n\n")

