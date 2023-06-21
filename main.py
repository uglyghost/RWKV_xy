import os
import logging
import types
import numpy as np
import torch
from src.utils import CustomDataset
from src.binidx import MMapIndexedDataset

# Configure the numpy print options and logging details.
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

local_env = os.environ.copy()
local_env["PATH"]=r"C:\Users\gy\.conda\envs\torch\Scripts;" + local_env["PATH"]
os.environ.update(local_env)



# Set environment variables.
EXPRESS_PILE_MODE = False
EXPRESS_PILE_MODEL_NAME = 'RWKV-4-Pile-169M-20220807-8023'
EXPRESS_PILE_MODEL_TYPE = 'RWKV-4-Pile-169M'
datafile = "./data/enwik8"
datafile_encoding = 'utf-8'
os.environ['VOCAB_SIZE'] = '50277' if EXPRESS_PILE_MODE else '0'
os.environ['RWKV_NUM_GPUS'] = '1'
# os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_FLOAT_MODE'] = 'fp16'
os.environ['RWKV_DEEPSPEED'] = '0' if int(os.environ['RWKV_NUM_GPUS']) == 1 else '1'
os.environ['USE_WANDB'] = '0'


# print(os.environ['PATH'])
# exit()

# Set model details.
EPOCH_BEGIN = 0
LOAD_MODEL = EXPRESS_PILE_MODE
n_layer = 12 if EXPRESS_PILE_MODE and EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-169M' else 6
n_embd = 768 if EXPRESS_PILE_MODE and EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-169M' else 512
ctx_len = 1024
model_type = 'RWKV'

# Set batch size and learning rate.
# batch_size = 12 * int(os.environ['RWKV_NUM_GPUS'])
batch_size = 1
lr_init = 2e-5 if EXPRESS_PILE_MODE and EXPRESS_PILE_MODEL_TYPE == 'RWKV-4-Pile-169M' else 8e-4
lr_final = 1e-5
n_epoch = 100000 if EXPRESS_PILE_MODE else 500
epoch_length_fixed = (10000 // batch_size) * batch_size
epoch_save_frequency = 10
epoch_save_path = 'trained-'
betas = (0.9, 0.999) if EXPRESS_PILE_MODE else (0.9, 0.99)
eps = 1e-8
num_workers = 1

# Set other variables.
NUM_GPUS = int(os.environ['RWKV_NUM_GPUS'])
os.environ['RWKV_LOAD_MODEL'] = str(LOAD_MODEL)
MODEL_NAME = EXPRESS_PILE_MODEL_NAME if EXPRESS_PILE_MODE else epoch_save_path + str(EPOCH_BEGIN)
warmup_tokens = 50 * ctx_len * batch_size // NUM_GPUS if LOAD_MODEL and EPOCH_BEGIN > 0 else 0

# Set the cuDNN benchmark and tf32 options.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = os.environ['RWKV_FLOAT_MODE'] != 'fp32'

# Load the data. 加载数据，随机取
print(f'loading {datafile_encoding} data... ' + datafile)
if datafile_encoding == 'binidx':
    train_dataset = CustomDataset(MMapIndexedDataset(datafile), ctx_len, epoch_length_fixed)
elif datafile_encoding == 'numpy':
    train_dataset = CustomDataset(np.load(datafile).astype('int'), ctx_len, epoch_length_fixed)
else:
    train_dataset =CustomDataset(open(datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed)

import os
from src.trainer import Trainer, TrainerConfig
import types
from pytorch_lightning.strategies import DeepSpeedStrategy

# Define the deepspeed configuration.
DEEPSPEED_CFG = {
    "zero_allow_untested_optimizer": True,
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "allgather_bucket_size": 200000000,
        "reduce_bucket_size": 200000000,
        "sub_group_size": 1000000000000
    },
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "synchronize_checkpoint_boundary": False
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "single_submit": False,
        "overlap_events": True,
        "thread_count": 1
    },
    "gradient_clipping": 1.0,
    "gradient_accumulation_steps": 1,
}

def set_deepspeed_config(num_gpus, float_mode):
    """
    Adjusts the deepspeed configuration based on the number of GPUs and the desired floating point precision mode.
    """
    if num_gpus == 1:
        DEEPSPEED_CFG['zero_optimization'] = {
            "stage": 1,
            "contiguous_gradients": False,
            "overlap_comm": False,
            "allgather_partitions": False,
            "reduce_scatter": False,
            "allgather_bucket_size": 200000000,
            "reduce_bucket_size": 200000000,
            "sub_group_size": 1000000000000
        }

    if float_mode == 'fp16':
        DEEPSPEED_CFG["fp16"] = {
            "fp16": True,
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 12,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    elif float_mode == 'bf16':
        DEEPSPEED_CFG["bf16"] = {
            "enabled": True
        }
    return DEEPSPEED_CFG


def create_trainer(num_gpus, float_mode, deepspeed):
    """
    Creates the trainer based on the number of GPUs, the desired floating point precision mode, and whether to use deepspeed.
    """
    precision_modes = {"fp16": 16,
                       "bf16": 'bf16',
                       "32": 32}
    if deepspeed:
        DEEPSPEED_CFG = set_deepspeed_config(num_gpus, float_mode)
        return Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=num_gpus, accelerator="gpu", precision=precision_modes[float_mode])
    else:
        return Trainer(devices=num_gpus, accelerator="gpu", precision=precision_modes[float_mode])


# Begin main execution.
if __name__ == '__main__':
    # Print some information about the model.
    print('\nmodel', model_type, os.environ['RWKV_FLOAT_MODE'], 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, '\n')

    # Define the training configuration.
    tconf = TrainerConfig(
        model_type=model_type,
        max_epochs=n_epoch,
        batch_size=batch_size,
        learning_rate=lr_init,
        lr_decay=True,
        lr_final=lr_final,
        betas=betas,
        eps=eps,
        warmup_tokens=warmup_tokens,
        final_tokens=n_epoch * len(train_dataset) * ctx_len,
        num_workers=num_workers,
        epoch_save_frequency=epoch_save_frequency,
        epoch_save_path=epoch_save_path
    )

    # Define the model configuration.
    m_cfg = types.SimpleNamespace()
    m_cfg.model_type = model_type
    m_cfg.n_layer = n_layer
    m_cfg.n_embd = n_embd
    m_cfg.EPOCH_BEGIN = EPOCH_BEGIN
    m_cfg.LOAD_MODEL = LOAD_MODEL
    m_cfg.MODEL_NAME = MODEL_NAME

    # Determine whether to use deepspeed based on the environment variable.
    use_deepspeed = os.environ['RWKV_DEEPSPEED'] != '0'

    # Create the trainer.
    trainer = create_trainer(NUM_GPUS, os.environ['RWKV_FLOAT_MODE'], use_deepspeed)

    # If using deepspeed, print the configuration.
    if use_deepspeed:
        print(trainer._strategy.config)

    # Run the trainer.
    trainer.run(m_cfg, train_dataset, None, tconf)
