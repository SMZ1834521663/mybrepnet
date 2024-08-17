SCENES=(377 386 387 392 393 394)
N_GPU=6
GPUS=(0 2 3 4 5 6)
mkdir train_log
for ((JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++)) 
do
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    GPU=${GPUS[$GPU_ID]}
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    nohup python train_net.py exp_name=zju_${SCENE}_rgb rgb_only=True subject=zju${SCENE} gpus=${GPU} ratio=0.5 resume=False > train_log/${SCENE}.log ratio=1 iterations=14000 &
done

# SCENES=(377 386 387 392 393 394)
# N_GPU=3
# GPUS=(0 1 3)
# mkdir train_log
# for ((JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++)) 
# do
#     GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
#     GPU=${GPUS[$GPU_ID]}
#     SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
#     nohup python train_net.py exp_name=zju_${SCENE}_0.5_rgb rgb_only=True subject=zju${SCENE} gpus=${GPU} resume=False >> train_log/${SCENE}.log ratio=0.5 iterations=14000 &
# done



