SCENES=(female-3-casual female-4-casual male-3-casual male-4-casual)
N_GPU=4
GPUS=(0 1 2 3)
mkdir train_log
for ((JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++)) 
do
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    GPU=${GPUS[$GPU_ID]}
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    nohup python train_net.py exp_name=${SCENE} ratio=1 subject=${SCENE} gpus=${GPU} snapshot=True resume=False iterations=14000 > log/train_log/${SCENE}.log &
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
#     nohup python train_net.py exp_name=zju_${SCENE}_0.5 subject=zju${SCENE} gpus=${GPU} resume=False >> train_log/${SCENE}.log ratio=0.5 iterations=14000 &
# done

