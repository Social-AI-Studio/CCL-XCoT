import logging

from packaging import version

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os

import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.strategies.ddp import DDPStrategy
import inspect
from pl_args import add_model_args, add_pl_args, add_program_args
from pl_data import LitContraCLMDataModule
from pl_model import LitContraCLM
from utils import (CheckpointEveryNSteps, GPUtilCallback, get_loss_func,
                   setup_log_path)

def filter_trainer_args(args):
    args_dict = vars(args)

    valid_params = inspect.signature(pl.Trainer).parameters

    filtered_args = {key: value for key, value in args_dict.items() if key in valid_params}
    return filtered_args
def main():
    assert version.parse(transformers.__version__) >= version.parse(
        '4.21.0.dev0'), "transformers version not supported"  # critical for CodeGen
    parser = add_program_args()
    parser = add_model_args(parser)
    parser = add_pl_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    num_nodes = 1

    args.train_batch_size = args.train_batch_size // (args.devices * num_nodes)
    args.valid_batch_size = args.valid_batch_size // (args.devices * num_nodes)
    logger.info(f'{args.train_batch_size=} {args.valid_batch_size=}')

    args.val_check_interval *= args.accumulate_grad_batches

    # Load Data
    data = LitContraCLMDataModule(
        args.expt_prefix,
        args.train_datadir, 
        args.valid_datadir, 
        args.train_batch_size, 
        args.valid_batch_size,
        num_workers=args.num_workers
    )
    data.setup()
    args.num_training_examples = len(data.train_dataloader())


    expt_name = setup_log_path(args, num_nodes)
    loggers = []
    loggers.append(pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, args.log_dir), name=expt_name))

    plugins = []
    callbacks = [LearningRateMonitor(logging_interval='step')]
    if args.debug_cuda_mem:
        callbacks.append(GPUtilCallback())

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="Valid/Loss/MLE",
        mode="min",
        every_n_train_steps=args.save_step_frequency
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(CheckpointEveryNSteps(save_step_frequency=5000))

    logger.info('Initializing PL Trainer...')
    custom_trainer_kwargs = {
        'callbacks': callbacks,
        'logger': loggers,
        'strategy': DeepSpeedStrategy(config=args.ds_config) \
            if args.use_deepspeed else DDPStrategy(find_unused_parameters=False),
        'num_nodes': num_nodes,
        'plugins': plugins
    }
    filtered_args = filter_trainer_args(args)
    #trainer = pl.Trainer(
    #    strategy="deepspeed_stage_2_offload",
    #    **filtered_args,
    #    **custom_trainer_kwargs)
    trainer = pl.Trainer(
        **filtered_args,
        **custom_trainer_kwargs)
    #trainer = trainer(**filtered_args, **custom_trainer_kwargs)
    logger.warning(f'{trainer.__dict__=}')
    # Initialize the Contrastive Learning Objectives
    loss_func_tok, loss_func_seq = get_loss_func(args, args.pad_token_id)

    # PL model
    pl_model = LitContraCLM(args,
                            loss_func_tok=loss_func_tok,
                            loss_func_seq=loss_func_seq,
                            num_nodes=num_nodes)

    print(os.path.join(args.default_root_dir, args.log_dir, expt_name, "last.ckpt"))
    # training
    trainer.fit(pl_model, data)

    # save ckpt at the last step
    if trainer.is_global_zero or args.use_deepspeed:
        # save_checkpoint on all rank with deepspeed to avoid hang
        # https://github.com/microsoft/DeepSpeed/issues/797
        save_path = os.path.join(args.default_root_dir, args.log_dir, expt_name, "last.ckpt")
        logger.info(f"Saving model to {save_path}")
        trainer.save_checkpoint(save_path, weights_only=True)
        logger.info("Finished saving")
        if args.use_deepspeed:
            # Avoid checkpoint corruption if node 0 exits earlier than other
            # nodes triggering termination of other nodes
            torch.distributed.barrier()


if __name__ == '__main__':
    main()
