SCENES=(377 386 387 392 393 394)
N_GPU=6
GPUS=(0 1 3 4 5 6)
mkdir log/train_log
for ((JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++)) 
do
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    GPU=${GPUS[$GPU_ID]}
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    nohup python animate.py mod=test exp_name=zju${SCENE} test_novel_pose=True subject=zju${SCENE}_animate  gpus=${GPU} ratio=0.5 gpus=0 animate=True &
done
