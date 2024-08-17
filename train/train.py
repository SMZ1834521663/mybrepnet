import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
import time

from models.brepnet import BRepNet
import utils.data_utils as data_utils 

#####pyOCC库绑定到原C++的OCC库：https://github.com/tpaviot/pythonocc-core
def my_save_results(log_dir, opts, results,epoch):
    output_file = log_dir / "test_results_{}.json".format(epoch)
    options_dict = {}
    for opt in vars(opts):
        options_dict[opt] = getattr(opts, opt)

    data = {
        "options": options_dict,
        "results": results
    }
    data_utils.save_json_data(output_file, data)

def save_results(log_dir, opts, results):
    output_file = log_dir / "test_results.json"
    options_dict = {}
    for opt in vars(opts):
        options_dict[opt] = getattr(opts, opt)

    data = {
        "options": options_dict,
        "results": results
    }
    data_utils.save_json_data(output_file, data)


def do_training(opts):

    # brepnet = BRepNet(opts)
    brepnet = BRepNet.load_from_checkpoint("/data/smz24/mybrepnet2/logs/0816/142223/checkpoints/epoch=8-step=962.ckpt", opts=opts)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="validation/loss",
        mode="min"
    )

    # Define a path to save the logs which is based on date and time
    # The dir will look something like this
    # logs/0430/123103/checkpoints
    # logs/0430/123103/hparams.yaml
    # logs/0430/123103/events.out.tfevents.xxxx
    month_day = time.strftime('%m%d')
    hour_min_second = time.strftime('%H%M%S')
    tb_logger = pl_loggers.TensorBoardLogger(
        opts.log_dir, 
        name = month_day,
        version = hour_min_second
    )

    log_dir = Path(tb_logger.log_dir)
    
    print(" ")
    print(" ")
    print("--------------------------------------------------------------------------")
    print("BRepNet: A topological message passing system for solid models")
    print(" ")
    print(f"Logs written to {opts.log_dir}/{month_day}/{hour_min_second}")
    print("To monitor the loss, accuracy and IoU use")
    print(" ")
    print("tensorboard --logdir logs")
    print(" ")
    print("The trained model with the best validation loss will be written to")
    print(f"{log_dir}/checkpoints")
    print(" ")

    # Create the trainer
    trainer = Trainer.from_argparse_args(
        opts, 
        callbacks=[checkpoint_callback], 
        logger=tb_logger,
        gpus=1
    )

    print("Starting training")
    trainer.fit(brepnet)
    print("End training")
    # brepnet(1,2,3,4,5,6,7,8,9)
    # Test using the best checkpoint (which the trainer will keep track of)
    test_results = trainer.test()
    print(f"Logs written to {log_dir}")
    save_results(log_dir, opts, test_results)
    print("End testing")

    output = {
        "month_day": month_day,
        "hour_min_second": hour_min_second,
        "test_results": test_results
    }

    # Return the output
    return output


if __name__ == '__main__':
    #0814151633
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BRepNet.add_model_specific_args(parser)
    # opts = parser.parse_args()
    opts = parser.parse_args(["--dataset_file","/data/public/smz24/fusion360_gallery/s2.0.1/processed/dataset.json","--dataset_dir","/data/public/smz24/fusion360_gallery/s2.0.1/processed/","--max_epochs","10"])
    do_training(opts)

    

    
