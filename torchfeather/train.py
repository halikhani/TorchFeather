import contextlib
import os
import signal
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import timedelta
from typing import Any, cast

import torch
from loguru import logger
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record

from torchfeather.components.checkpoint import CheckpointManager
from torchfeather.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchfeather.components.loss import (
    IGNORE_INDEX,
    LossFunction,
    build_cross_entropy_loss,
)
from torchfeather.components.lr_scheduler import (
    LRSchedulersContainer,
    build_lr_schedulers,
)
from torchfeather.components.metrics import (
    MetricsProcessor,
    collect_parameter_norm_metrics,
)
from torchfeather.components.optimizer import (
    OptimizersContainer,
    build_optimizers_with_moe_load_balancing,
)
from torchfeather.components.tokenizer import (
    DeepSeekV3Tokenizer,
)
from torchfeather.config import TORCH_DTYPE_MAP, JobConfig
from torchfeather.config.default_configs import (
    get_config,
)
from torchfeather.config.job_config import Parallelism
from torchfeather.datasets.hf_datasets import build_hf_dataloader
from torchfeather.distributed import ParallelDims
from torchfeather.distributed import utils as dist_utils
from torchfeather.distributed.pipeline_parallel import pipeline_llm
from torchfeather.model.model import DeepSeekV3Model
from torchfeather.model.parallelize import parallelize_deepseekv3
from torchfeather.tools import device_utils, utils
from torchfeather.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


class Trainer(Stateful):
    job_config: JobConfig
    parallel_dims: ParallelDims

    tokenizer: DeepSeekV3Tokenizer
    dataloader: BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: Callable[..., contextlib.AbstractContextManager]
    maybe_enable_amp: contextlib.AbstractContextManager
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    step: int
    ntokens_seen: int

    def __init__(self, job_config: JobConfig):
        self.job_config = job_config

        device_module, device_type = (
            device_utils.device_module,
            device_utils.device_type,
        )
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # init distributed and build meshes
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=job_config.comm.init_timeout_seconds),
        )
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = self._create_parallel_dims(
            parallelism_config, world_size
        )

        _ = parallel_dims.world_mesh
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            dp_degree, dp_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0


        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )

        # build tokenizer and dataloader
        self.tokenizer = DeepSeekV3Tokenizer(job_config.model.hf_assets_path)

        self.dataloader = build_hf_dataloader(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=self.tokenizer,
            job_config=job_config,
        )

        model_args = job_config.model.args
        model_args.max_seq_len = job_config.training.seq_len
        # Build on the meta device
        with (
            torch.device("meta"),
            device_utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            model = DeepSeekV3Model(model_args)

        # why meta device?
        #         meta parameter
        #     |
        #     | shape exists, no data
        #     v
        # parallelize / shard model
        #     |
        #     v
        # to_empty(cuda)
        #     |
        #     | real GPU memory now allocated
        #     v
        # init_weights()
        #     |
        #     v
        # actual initialized GPU parameters


        self.metrics_processor = MetricsProcessor(job_config, parallel_dims)

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(f"Model total parameters: {model_param_count:,}")

        self.loss_fn = build_cross_entropy_loss(job_config)

        # verify batch sizes
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            global_batch_size = job_config.training.local_batch_size * dp_degree
        assert global_batch_size > 0, global_batch_size

        # This is when you specify the global batch size manually.
        # It is useful for validation of your config
        
        assert (
            global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0
        ), (global_batch_size, (job_config.training.local_batch_size * dp_degree))

        # calculate gradient accumulation steps
        self.gradient_accumulation_step = global_batch_size // (
            job_config.training.local_batch_size * dp_degree
        )

        assert self.gradient_accumulation_steps > 0

        init_device = device_type
        buffer_device = None

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = pipeline_llm(
                model,
                parallel_dims,
                job_config,
                self.device,
                model_args.n_layers,
                parallelize_deepseekv3,
                self.loss_fn,
            )

            # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)  # ty:ignore[call-non-callable]
                m.train()

        else:
            model = parallelize_deepseekv3(model, parallel_dims, job_config)

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

            



            
