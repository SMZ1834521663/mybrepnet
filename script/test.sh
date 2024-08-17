SCENES=(377 386 387 392 393 394)
N_GPU=6
GPUS=(0 1 3 4 5 6)
mkdir test_log
for ((JOB_COMPLETION_INDEX=0; JOB_COMPLETION_INDEX<${#SCENES[@]}; JOB_COMPLETION_INDEX++)) 
do
    GPU_ID=$(expr $JOB_COMPLETION_INDEX % $N_GPU)
    GPU=${GPUS[$GPU_ID]}
    SCENE=${SCENES[$JOB_COMPLETION_INDEX]}
    # echo novel_view >> test_log/${SCENE}.log &
    nohup python train_net.py exp_name=zju${SCENE} subject=zju${SCENE} gpus=${GPU} ratio=1 resume=False mod=test & #test_novel_pose=True & 
    # echo novel_pose: >> test_log/${SCENE}.log &
    # nohup python train_net.py exp_name=zju_${SCENE} subject=zju${SCENE} gpus=${GPU} resume=False mod=test test_novel_pose=True &
done
