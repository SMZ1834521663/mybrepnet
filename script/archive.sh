SCENES=(377 386 387 392 393 394)
N_GPU=3
GPUS=(0 1 3)
exp_name=(view)
mkdir archive/${exp_name}

mv data/result/deform/* archive/${exp_name}/
mv data/record/deform/*  archive/${exp_name}/
mv data/trained_model/deform/*  archive/${exp_name}/
mv test_log/  archive/${exp_name}/
mv train_log/  archive/${exp_name}/

