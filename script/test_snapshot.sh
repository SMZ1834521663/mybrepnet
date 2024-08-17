SCENES=(female-3-casual female-4-casual male-3-casual male-4-casual)
N_GPU=4
GPUS=(0 1 2 3)
mkdir train_log
for ((JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++)) 
do
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    GPU=${GPUS[$GPU_ID]}
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    nohup python train_net.py exp_name=${SCENE} subject=${SCENE} gpus=${GPU} snapshot=True iterations=14000 mod=test &
done
