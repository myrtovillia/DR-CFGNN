import subprocess
import sys

dataset_name = None
for arg in sys.argv[1:]:
    if arg.startswith("datasets="):
        dataset_name = arg.split("=", 1)[1]

DATASET_CLASSES = {    
    "ba_2motifs": [0, 1],
    "ba_2motifs_3class": [0, 1], # we do not want the class 2, it is an incomplete class
    "ba_3motifs": [0, 1, 2],
    "ba_4motifs": [0, 1, 2, 3],
    "bbbp": [0, 1],
    "graph_sst2": [0, 1],
    "twitter": [0, 1, 2],
    "graph_sst5": [0, 1, 2, 3, 4],
}
if dataset_name not in DATASET_CLASSES:
    raise ValueError(
        f"Dataset '{dataset_name}' not in DATASET_CLASSES. "
        f"Available: {list(DATASET_CLASSES.keys())}"
    )
predefined_classes = DATASET_CLASSES[dataset_name]



# -----------------------
# RANDOM
# -----------------------
cmd = [
    "python", "-m", "benchmarks.xgraph.DR_CFGNN",
    f"datasets={dataset_name}",
    "dr_cfgnn.run_dr_cfgnn=False" ,
    "dr_cfgnn.run_random_baseline=True"
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=False)


 
# -----------------------
# denoising_mode=none + one_hot=False 
# -----------------------
cmd = [
    "python", "-m", "benchmarks.xgraph.DR_CFGNN",
    f"datasets={dataset_name}",
    "denoising_mode=none",
    "one_hot_reconst=False",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=False)



# -----------------------
#  denoising_mode=none + one_hot=True + all predefined classes
# -----------------------
for cf_classes in predefined_classes:
    cmd = [
        "python", "-m", "benchmarks.xgraph.DR_CFGNN",
        f"datasets={dataset_name}",
        "denoising_mode=none",
        "one_hot_reconst=True",
        f"dr_cfgnn.predefined_cf_class={cf_classes}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)





# -----------------------
# denoising_mode=without_one_hot + one_hot=False 
# -----------------------
cmd = [
    "python", "-m", "benchmarks.xgraph.DR_CFGNN",
    f"datasets={dataset_name}",
    "denoising_mode=without_one_hot",
    "one_hot_reconst=False",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=False)




# -----------------------
# denoising_mode=without_one_hot + one_hot=True + all predefined classes
# -----------------------
for cf_classes in predefined_classes:
    cmd = [
        "python", "-m", "benchmarks.xgraph.DR_CFGNN",
        f"datasets={dataset_name}",
        "denoising_mode=without_one_hot",
        "one_hot_reconst=True",
        f"dr_cfgnn.predefined_cf_class={cf_classes}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)

