# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from aas import config as cfg
from aas.env import AASTrainer
from utils.log import print_info
import os

# Set trainer
trainer = AASTrainer(env_type='op', save_file_path=os.getenv('AAS_SAVE_FILE_PATH', None))
print_info('Env build ...')
print_info(f'Temporary training env file at {trainer.train_env.tmp_file_path}')
print_info(f'Temporary evaluation env file at {trainer.eval_env.tmp_file_path}')
print_info(f"Agent's network saved at {trainer.save_file_path}")
if cfg.logging:
    print_info(f'Logging to {trainer.file_logger.base_dir_path}')
# Print configuration
print_info('Configuration:')
print_info(cfg)

# Start training
print_info('Start training ... ')

trainer.train()

print_info('Training ended ... ')
