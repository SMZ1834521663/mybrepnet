

# python train_net.py exp_name=wo_everything ratio=0.5 subject=zju377 gpus=4 resume=False view_dependent=False pose_color=False pose_correction=False
# python train_net.py exp_name=wo_everything ratio=0.5 subject=zju377 gpus=4 resume=False view_dependent=False pose_color=False pose_correction=False mod=test check_ckpt=True

# python train_net.py exp_name=with_view ratio=0.5 subject=zju377 gpus=4 resume=False view_dependent=True pose_color=False pose_correction=False
# python train_net.py exp_name=with_view ratio=0.5 subject=zju377 gpus=4 resume=False view_dependent=True pose_color=False pose_correction=False mod=test check_ckpt=True

# python train_net.py exp_name=with_view ratio=0.5 subject=zju377 gpus=4 resume=False view_dependent=True pose_color=True pose_correction=False
# python train_net.py exp_name=with_view ratio=0.5 subject=zju377 gpus=4 resume=False view_dependent=True pose_color=True pose_correction=False mod=test check_ckpt=True

# python train_net.py exp_name=pose ratio=0.5 subject=zju377 gpus=4 resume=False  pose_correction=True
# python train_net.py exp_name=pose ratio=0.5 subject=zju377 gpus=4 resume=False  pose_correction=True mod=test check_ckpt=True


# python train_net.py exp_name=can_view ratio=0.5 subject=zju377 gpus=4 resume=False  
# python train_net.py exp_name=can_view ratio=0.5 subject=zju377 gpus=4 resume=False  mod=test check_ckpt=True

# python train_net.py exp_name=wopose_clr ratio=0.5 subject=zju377 gpus=4 resume=False pose_color=False 
python train_net.py exp_name=wopose_clr ratio=0.5 subject=zju377 gpus=4 resume=False pose_color=False mod=test check_ckpt=True

# python train_net.py exp_name=wo_txyz ratio=0.5 subject=zju377 gpus=4 resume=False pose_color=False txyz=False
python train_net.py exp_name=wo_txyz ratio=0.5 subject=zju377 gpus=4 resume=False pose_color=False txyz=False mod=test check_ckpt=True




