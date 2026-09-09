import torch
import copy
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmrotate.utils import find_latest_checkpoint
import torch.distributed as dist

@HOOKS.register_module()
class RestartOnNanHook(Hook):
    def __init__(self):
        self.initial_state = None

    def before_run(self, runner):
        # Cache initial weights to restart from scratch if needed
        self.initial_state = {
            'model': {k: v.cpu().clone() for k, v in runner.model.state_dict().items()},
            'optimizer': copy.deepcopy(runner.optimizer.state_dict())
        }

        # Monkey-patch runner.train to allow aborting the epoch without crashing the process
        original_train = runner.train
        def safe_train(*args, **kwargs):
            try:
                original_train(*args, **kwargs)
            except RuntimeError as e:
                if str(e) == "NaN_ABORT_EPOCH":
                    runner.logger.info("Epoch aborted successfully. The next loop will restart from the restored epoch 0-th batch.")
                else:
                    raise e
        runner.train = safe_train

    def after_train_iter(self, runner):
        # 1. Check for NaN locally
        is_nan = False
        for key, val in runner.outputs.items():
            if 'loss' in key and torch.isnan(val).any():
                is_nan = True
                runner.logger.warning(f"NaN detected in {key} on rank {runner.rank}!")
                break
                
        # 2. Synchronize the NaN flag across all GPUs (if distributed)
        rank, world_size = get_dist_info()
        
        is_nan_tensor = torch.tensor(int(is_nan), dtype=torch.int32, device='cuda')
        if world_size > 1:
            dist.all_reduce(is_nan_tensor, op=dist.ReduceOp.MAX)

        # 3. If any GPU got NaN, all GPUs execute the rollback
        if is_nan_tensor.item() > 0:
            if rank == 0:
                runner.logger.warning("NaN detected on at least one GPU! Synchronously restoring checkpoint...")
                
            latest_ckpt = find_latest_checkpoint(runner.work_dir)
            if latest_ckpt is not None:
                if rank == 0:
                    runner.logger.info(f"Resuming from {latest_ckpt}")
                # Rollback model and optimizer state in-memory
                runner.resume(latest_ckpt)
            else:
                if rank == 0:
                    runner.logger.warning("No checkpoint found. Restarting training from scratch...")
                if self.initial_state is not None:
                    # Restore cached initial state
                    runner.model.load_state_dict(self.initial_state['model'])
                    runner.optimizer.load_state_dict(self.initial_state['optimizer'])

                    # Reset counters
                    runner._epoch = 0
                    runner._iter = 0
                    runner._inner_iter = 0
            
            # Synchronously abort the current epoch loop
            raise RuntimeError("NaN_ABORT_EPOCH")
