# python train_net.py exp_name=base ratio=0.5 subject=zju377 gpus=0 resume=False
# python train_net.py exp_name=base ratio=0.5 subject=zju377 gpus=0 resume=False mod=test check_ckpt=True

python train_net.py exp_name=base2 ratio=0.5 subject=zju377 gpus=0 resume=False lambda_iso=10
python train_net.py exp_name=base2 ratio=0.5 subject=zju377 gpus=0 resume=False lambda_iso=10 mod=test check_ckpt=True

python train_net.py exp_name=base3 ratio=0.5 subject=zju377 gpus=0 resume=False lambda_iso=10 lambda_reg_res_rot=0.1
python train_net.py exp_name=base3 ratio=0.5 subject=zju377 gpus=0 resume=False lambda_iso=10 lambda_reg_res_scale=0.1 mod=test check_ckpt=True


python train_net.py exp_name=base3 ratio=0.5 subject=zju377 gpus=0 resume=False lambda_iso=10 lambda_reg_res_rot=0.1
python train_net.py exp_name=base3 ratio=0.5 subject=zju377 gpus=0 resume=False lambda_iso=10 lambda_reg_res_scale=0.1 mod=test check_ckpt=True