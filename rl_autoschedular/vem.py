from datetime import timedelta
from time import time
import torch
from rl_autoschedular import device
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.trajectory import TrajectoryData
from utils.config import Config
from utils.file_logger import FileLogger
from utils.log import print_error, print_info


def vem_fit(trajectory: TrajectoryData, model: Model, optimizer: torch.optim.Optimizer):
    """Fit policy to offline data

    Args:
        trajectory (Trajectory): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
    """
    fl = FileLogger()
    cfg = Config()

    trajectory.update_attributes(model)

    vem_start = time()
    data_loader = trajectory.loader(cfg.vem_batch_size, 1)
    for _ in range(cfg.vem_epochs):
        for batch in data_loader:
            batch = [e.to(device, non_blocking=True) for e in batch]
            (
                _, _,
                obs,
                _, _, _, _,
                values,
                _, _, _,
                returns,
                _,
            ) = batch
            with torch.enable_grad():
                new_values = model.value_model(obs)

                loss = model.value_model.loss(new_values, values, returns, 0.9)

            optimizer.zero_grad()
            try:
                loss.backward()
                clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                for target_param, param in zip(model.target_value_model.parameters(), model.value_model.parameters()):
                    target_param.data.copy_((1 - cfg.vem_soft_update) * target_param.data + cfg.vem_soft_update * param.data)
            except Exception as e:
                print_error(
                    'Error during Value update\n'
                    f'Error: {e}'
                )

            # Logging
            fl['train_vem/value_loss'].append(loss.item())
            fl['train_vem/value_clip_factor'].append(clip_factor.item())

    for target_param, param in zip(model.target_value_model.parameters(), model.value_model.parameters()):
        param.data.copy_(target_param.data)

    trajectory.update_attributes(model)

    data_loader = trajectory.loader(cfg.vem_batch_size, 1)
    for _ in range(cfg.vem_epochs):
        for batch in data_loader:
            batch = [e.to(device, non_blocking=True) for e in batch]
            (
                _,
                actions_index,
                obs,
                _, _, _, _,
                values,
                _, _,
                off_policy_rates,
                returns,
                advantages,
            ) = batch
            max_abs_adv = advantages.abs().max()
            if cfg.normalize_adv == 'standard' and advantages.size(0) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif cfg.normalize_adv == 'max-abs' and max_abs_adv > 0:
                advantages = advantages / max_abs_adv

            with torch.enable_grad():
                actions_log_p, new_values, _ = model(obs, actions_index)

                loss, _ = model.policy_model.loss(actions_log_p, actions_log_p, off_policy_rates, advantages)

            optimizer.zero_grad()
            try:
                loss.backward()
                clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            except Exception as e:
                print_error(
                    'Error during PPO update\n'
                    f'Error: {e}'
                )

            # Logging
            fl['train_vem/policy_loss'].append(loss.item())
            fl['train_vem/policy_clip_factor'].append(clip_factor.item())
    vem_end = time()
    print_info(f"VEM fit in {timedelta(seconds=vem_end - vem_start)}")
