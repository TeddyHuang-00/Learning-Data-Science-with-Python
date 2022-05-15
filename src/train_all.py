import os
import time


# Scripts & models to be run in one loop
task_list = [
    "CNN.py",
    "Adversarial.py",
    "Quantization.py",
    "Regularization.py",
    "RandomSelfEnsemble.py",
    "Combination.py",
    "PGD.py",
]


# This will sequentially execute all the files in task_list
def train_one_loop():
    global task_list

    for task in task_list:
        start = time.time()
        print("-" * 25 + f"Running {task}" + "-" * 25)
        os.system("python3 " + task)
        print(f"Time elapsed: {time.time() - start:.3f} sec training {task}")

    print("-" * 25 + "Summary" + "-" * 25)
    os.system("tail -n 3 ./log/*log")
    with open("./log/summary.txt", "w") as f:
        with open("./params.py", "r") as param:
            lines = [line for line in param.readlines() if not line.startswith("#")]

        print("-" * 25 + "Paramters" + "-" * 25)
        print("".join(lines))

        f.write("-" * 25 + "Paramters" + "-" * 25 + "\n")
        f.write("".join(lines))
        f.write("\n\n")
    os.system("tail -n 3 ./log/*log >> ./log/summary.txt")


def set_seed(seed=0):
    with open("./params.py", "r") as f:
        lines = f.readlines()
    with open("./params.py", "w") as f:
        f.write("".join([l if "SEED" not in l else f"SEED = {seed}\n" for l in lines]))


def set_epsilon(epsilon=8):
    with open("./params.py", "r") as f:
        lines = f.readlines()
    with open("./params.py", "w") as f:
        f.write(
            "".join(
                [
                    l if "ATTACK_EPS" not in l else f"ATTACK_EPS = {epsilon} / 255\n"
                    for l in lines
                ]
            )
        )


# This sets the seeds for the loops and then calls the former function to run a loop
def train_multiple_loops(seeds, epsilons):
    for seed in seeds:
        set_seed(seed)
        for epsilon in epsilons:
            set_epsilon(epsilon)
            train_one_loop()
            # Also collect important outputs (log, loss history and figures)
            timestamp = time.strftime("%m-%d-%H-%M", time.localtime())
            os.mkdir(f"./log/{timestamp}")
            os.system(f"mv ./log/*.* ./log/{timestamp}")
            os.system(f"cp ./fig/PGD.pdf ./log/{timestamp}")
            # removes adv models for a new epsilon
            os.system(f"rm ./model/*Adv*.ckpt")
        # removes all checkpoints for a new seed
        os.system("rm ./model/*.ckpt")


if __name__ == "__main__":
    os.system("rm ./model/*.ckpt")
    train_multiple_loops(list(range(0, 10)), list(range(4, 18, 4)))
