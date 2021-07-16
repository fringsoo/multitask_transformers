import json
import logging
import os
import random
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm.auto import tqdm, trange

from transformers.data.data_collator import DataCollator, DefaultDataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from transformers.training_args import TrainingArguments, is_tpu_available

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for the first one (locally) to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

class Evaluator():
    """
    Evaluator does evaluation for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
        alternate = False,
        task = None
    ):
        """
        Evaluater is a simple eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model
        self.args = args
        self.alternate = alternate
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DefaultDataCollator()
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.args.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        # if is_wandb_available():
        #     self._setup_wandb()
        # else:
        #     logger.info(
        #         "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
        #         "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
        #     )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_local_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        self.task = task
        if self.alternate and self.task != "Alttask":
            logger.warning("alternate=True can be consistent only with task Alttask")

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        sampler = get_tpu_sampler(eval_dataset) if is_tpu_available() else None

        batch_size = 1 if self.alternate else self.args.eval_batch_size
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.data_collator.collate_batch,
        )

        if is_tpu_available():
            data_loader = pl.ParallelLoader(data_loader, [self.args.device]).per_device_loader(self.args.device)

        return data_loader
    
    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
        # keep track of model topology and gradients
        if os.getenv("WANDB_WATCH") != "false":
            wandb.watch(
                self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
            )
    
    def num_examples(self, dataloader: Union[DataLoader, "pl.PerDeviceLoader"]) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        if is_tpu_available():
            assert isinstance(dataloader, pl.PerDeviceLoader)
            return len(dataloader._loader._loader.dataset)
        else:
            return len(dataloader.dataset)
    
    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
        if is_wandb_available():
            wandb.log(logs, step=self.global_step)
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def is_local_master(self) -> bool:
        if is_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0
    
    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> PredictionOutput:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        # self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        # multi-gpu eval
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model
        model.to(self.args.device)

        if is_tpu_available():
            batch_size = dataloader._loader._loader.batch_size
        else:
            batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds_t1: np.ndarray = None
        preds_t2: np.ndarray = None
        label_ids_t1: np.ndarray = None
        label_ids_t2: np.ndarray = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs.get(k) is not None for k in ["labels", "labels_t1", "labels_t2", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)

                if has_labels:
                    if not self.task: 
                        if self.alternate:
                            step_eval_loss, logits, task = outputs[:3]
                        else:
                            step_eval_loss, logits_t1, logits_t2 = outputs[:3]
                        eval_losses += [step_eval_loss.mean().item()]
                    else:
                        if self.task == "ADtask": #"Multitask,ADtask,SNStask"
                            step_eval_loss, logits = outputs[:2]
                        elif self.task == "SNStask":
                            step_eval_loss, logits = outputs[:2]
                        elif self.task == "Multitask" :
                            pass
                else:
                    logits = outputs[0]

            if self.alternate:
                if not prediction_loss_only:
                    if task==0:
                        if preds_t1 is None:
                            preds_t1 = logits.detach().cpu().numpy()
                        else:
                            preds_t1 = np.append(preds_t1, logits.detach().cpu().numpy(), axis=0)
                        if inputs.get("labels") is not None:
                            if label_ids_t1 is None:
                                label_ids_t1 = inputs["labels"].detach().cpu().numpy()
                            else:
                                label_ids_t1 = np.append(label_ids_t1, inputs["labels"].detach().cpu().numpy(), axis=0)

                    elif task==1:
                        if preds_t2 is None:
                            preds_t2 = logits.detach().cpu().numpy()
                        else:
                            preds_t2 = np.append(preds_t2, logits.detach().cpu().numpy(), axis=0)
                        if inputs.get("labels") is not None:
                            if label_ids_t2 is None:
                                label_ids_t2 = inputs["labels"].detach().cpu().numpy()
                            else:
                                label_ids_t2 = np.append(label_ids_t2, inputs["labels"].detach().cpu().numpy(), axis=0)

            else:
                if not prediction_loss_only:
                    if not self.task:
                        if preds_t1 is None or preds_t2 is None:
                                preds_t1 = logits_t1.detach().cpu().numpy()
                                preds_t2 = logits_t1.detach().cpu().numpy()
                        else:
                            preds_t1 = np.append(preds_t1, logits_t1.detach().cpu().numpy(), axis=0)
                            preds_t2 = np.append(preds_t2, logits_t2.detach().cpu().numpy(), axis=0)
                    else: ## for case that task is specified
                        if self.task == 'ADtask':
                            if preds_t2 is None:
                                preds_t2 = logits.detach().cpu().numpy()
                            else:
                                preds_t2 = np.append(preds_t2, logits.detach().cpu().numpy(), axis=0)
                        elif self.task == "SNStask":
                            if preds_t1 is None:
                                preds_t1 = logits.detach().cpu().numpy()
                            else:
                                preds_t1 = np.append(preds_t1, logits.detach().cpu().numpy(), axis=0)
                        elif self.task == "Multitask" :
                            if preds_t1 is None or preds_t2 is None:
                                preds_t1 = logits_t1.detach().cpu().numpy()
                                preds_t2 = logits_t1.detach().cpu().numpy()
                            else:
                                preds_t1 = np.append(preds_t1, logits_t1.detach().cpu().numpy(), axis=0)
                                preds_t2 = np.append(preds_t2, logits_t2.detach().cpu().numpy(), axis=0)
                    ## all true labels are treated the same in case where self.alternate is false
                    if inputs.get("labels_t1") is not None:
                        if label_ids_t1 is None or label_ids_t2 is None:
                            label_ids_t1 = inputs["labels_t1"].detach().cpu().numpy()
                            label_ids_t2 = inputs["labels_t2"].detach().cpu().numpy()
                        else:
                            label_ids_t1 = np.append(label_ids_t1, inputs["labels_t1"].detach().cpu().numpy(), axis=0)
                            label_ids_t2 = np.append(label_ids_t2, inputs["labels_t2"].detach().cpu().numpy(), axis=0)

        # if is_tpu_available() and preds is not None and label_ids is not None:
        #     # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
        #     preds = xm.mesh_reduce("eval_preds", preds, np.concatenate)
        #     label_ids = xm.mesh_reduce("eval_out_label_ids", label_ids, np.concatenate)

        metrics = {}
        if self.compute_metrics is not None:
            if preds_t1 is not None and label_ids_t1 is not None:
                metrics["task 1"] = self.compute_metrics(EvalPrediction(predictions=preds_t1, label_ids=label_ids_t1))
            if preds_t2 is not None and label_ids_t2 is not None:
                metrics["task 2"] = self.compute_metrics(EvalPrediction(predictions=preds_t2, label_ids=label_ids_t2))

        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        
        if not self.task or self.task =='Multitask':
            return (PredictionOutput(predictions=preds_t1, label_ids=label_ids_t1, metrics=metrics), PredictionOutput(predictions=preds_t2, label_ids=label_ids_t2, metrics=metrics))
        elif self.task =='ADtask':
            return (PredictionOutput(predictions=preds_t2, label_ids=label_ids_t2, metrics=metrics)) 
        elif self.task =='SNStask':
            return (PredictionOutput(predictions=preds_t1, label_ids=label_ids_t1, metrics=metrics)) 

    