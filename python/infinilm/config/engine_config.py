from dataclasses import dataclass
from typing import Optional

from infinilm.config.kv_transfer import KVTransferConfig


@dataclass
class EngineConfig:
    """Configuration for LLM Engine.

    Attributes:
        model_path: Path to the model directory.
        draft_model_path: Optional Eagle/MTP draft model directory.
        num_draft_tokens: Number of Eagle draft tokens to verify per step.
        device: Device type string ('cpu', 'cuda', 'mlu', etc.).
        dtype: Data type string ('float16', 'bfloat16', 'float32').
        tensor_parallel_size: Number of devices for tensor parallelism.
        pipeline_parallel_size: Number of pipeline stages.
        pipeline_parallel_stage: Pipeline stage index for this engine.
        master_addr: Address used to bootstrap distributed communication.
        master_port: TCP port used to bootstrap distributed communication.
        moe_ep_backend: MoE expert-parallel backend.
        moe_ep_size: MoE expert-parallel size.
        cache_type: Cache type ('paged' or 'static').
        max_batch_size: Maximum batch size for inference (only for paged cache).
        static_cache_capacity: Physical static KV rows (1 or 2); prefills are separate and decode can batch both rows.
        max_tokens: Default maximum tokens to generate.
        num_blocks: Number of KV cache blocks (only for paged cache).
        block_size: Size of each KV cache block (only for paged cache).
        max_cache_len: Maximum sequence length (only for static cache).
        enable_prefix_caching: Whether to reuse KV cache across requests.
        temperature: Default sampling temperature.
        top_p: Default top-p sampling parameter.
        top_k: Default top-k sampling parameter.
        enable_graph: Whether to enable graph compiling.
        attn_backend: Attention backend to use ('default', 'flash-attn').
        use_mla: Whether to use DeepSeek V2 MLA attention when supported.
        weight_load_mode: Weight loading mode across tensor-parallel workers.
        skip_load: Whether to skip loading model weights (for testing).
        enable_sla_constraint: Whether to enable SLA-constrained dynamic
            batching (paper Algorithm 2). Paged cache only.
        sla_latency: Decode TBT SLA target in seconds (D_SLA).
        sla_epsilon: Tolerance band around the SLA target (epsilon_D).
        batch_size_lower: Lower bound of the dynamic batch search range (B_min).
        batch_size_upper: Upper bound of the search range (B_max). 0 follows
            max_batch_size. Must not exceed max_batch_size (CUDA graph limit).
        sla_alpha: Minimum gap kept between the temporary bounds (alpha).
        sla_sigma: Correction step applied to the temporary bounds (delta).
        sla_history_window: Number of recent decode steps kept for statistics.
        sla_queue_refill: Inside the SLA band, raise the lower bound of the
            search range to (running + waiting) so a burst backlog cannot pin
            the cap to the running count and starve admission.
        sla_mem_bound_floor: Never let the memory bound (Algorithm 1) return
            less than B_min, so prefix-cache occupancy cannot collapse the cap
            to the running count and weld admission shut.
        sla_mem_bound_accurate: Estimate blocks_per_seq from the blocks each
            running request actually holds, instead of the pool-wide
            used/running ratio (which also counts queued requests' reserved
            blocks and prefix-cache blocks).
        sla_admission_fallback: When the per-step admission budget computes to
            0 but the pool still has free blocks, admit what those blocks can
            hold. Budgets already positive are left untouched.
        max_prefill_starvation_steps: Force one decode step after this many
            consecutive prefill-only steps (0 = disabled).
        enable_memory_aware_batching: Whether to enable the memory-aware batch
            bound (paper Algorithm 1).
        memory_usage_factor: Fraction of total KV blocks the memory bound may
            use (epsilon_M).
    """

    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: int = 4
    device: str = "cuda"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    pipeline_parallel_stage: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    moe_ep_backend: str = "disabled"
    moe_ep_size: int = 1
    cache_type: str = "paged"  # "paged" or "static"
    max_batch_size: int = 16
    max_tokens: int = 4096
    num_blocks: int = 512
    block_size: int = 256
    max_cache_len: int = 4096
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 1
    enable_graph: bool = False
    attn_backend: str = "default"
    use_mla: bool = False
    pre_transpose: bool = False
    weight_load_mode: str = "async"
    skip_load: bool = False
    kv_transfer_config: Optional[KVTransferConfig] = None
    enable_prefix_caching: bool = True
    static_cache_capacity: int = 1
    # --- SLA-constrained dynamic batching (paper Algorithm 1 + 2) ---
    # All defaults keep the feature disabled: behaviour with the flags off is
    # byte-for-byte identical to the stock static-batch scheduler.
    enable_sla_constraint: bool = False
    sla_latency: float = 0.05  # D_SLA: decode TBT target (seconds)
    sla_epsilon: float = 0.005  # epsilon_D: tolerance band
    batch_size_lower: int = 1  # B_min
    batch_size_upper: int = 0  # B_max; 0 = follow max_batch_size
    sla_alpha: int = 8  # alpha
    sla_sigma: int = 1  # delta
    sla_history_window: int = 100
    sla_queue_refill: bool = False  # see docstring
    sla_mem_bound_floor: bool = False  # see docstring
    sla_mem_bound_accurate: bool = False  # see docstring
    sla_admission_fallback: bool = False  # see docstring
    max_prefill_starvation_steps: int = 0
    enable_memory_aware_batching: bool = False  # paper Algorithm 1
    memory_usage_factor: float = 0.95  # epsilon_M
    # Measurement only: record decode step latency / batch size even when the
    # control loop is off (so a static-batch control arm can report the same
    # numbers). Does not affect scheduling in any way.
    enable_batch_stats: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.static_cache_capacity) is not int
            or self.static_cache_capacity < 1
        ):
            raise ValueError("static_cache_capacity must be a positive integer")
        if self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        if not 0 <= self.pipeline_parallel_stage < self.pipeline_parallel_size:
            raise ValueError(
                "pipeline_parallel_stage must be in [0, pipeline_parallel_size)"
            )
        if not 1 <= self.master_port <= 65535:
            raise ValueError("master_port must be in [1, 65535]")

        if self.weight_load_mode not in {"async", "sync"}:
            raise ValueError("weight_load_mode must be either 'async' or 'sync'")

        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector
            and self.cache_type != "paged"
        ):
            raise ValueError("kv_transfer_config requires cache_type='paged'")

        # --- SLA-constrained dynamic batching ---
        if (
            self.enable_sla_constraint or self.enable_memory_aware_batching
        ) and self.cache_type != "paged":
            raise ValueError(
                "enable_sla_constraint / enable_memory_aware_batching require "
                "cache_type='paged' (paged attention KV cache)"
            )

        if self.enable_sla_constraint:
            if self.sla_latency <= 0:
                raise ValueError("sla_latency must be > 0 (decode TBT SLA, seconds)")
            if self.sla_epsilon < 0:
                raise ValueError("sla_epsilon must be >= 0")
            if self.sla_epsilon >= self.sla_latency:
                raise ValueError(
                    "require sla_epsilon < sla_latency "
                    f"(got {self.sla_epsilon} / {self.sla_latency}); the SLA "
                    "tolerance band [D-epsilon, D+epsilon] would be empty"
                )
            if self.batch_size_lower < 1:
                raise ValueError("batch_size_lower must be >= 1")
            upper = self.batch_size_upper or self.max_batch_size
            if not self.batch_size_lower <= upper <= self.max_batch_size:
                raise ValueError(
                    "require batch_size_lower <= batch_size_upper <= "
                    f"max_batch_size (got {self.batch_size_lower} / {upper} / "
                    f"{self.max_batch_size}); batch_size_upper is bounded by "
                    "max_batch_size because CUDA graphs are only captured up "
                    "to --max-batch-size"
                )
            if self.sla_alpha < 1:
                raise ValueError("sla_alpha must be >= 1")
            if self.sla_sigma < 1:
                raise ValueError("sla_sigma must be >= 1")
            if self.sla_history_window < 1:
                raise ValueError("sla_history_window must be >= 1")
            if self.max_prefill_starvation_steps < 0:
                raise ValueError("max_prefill_starvation_steps must be >= 0")

        if not 0.0 < self.memory_usage_factor <= 1.0:
            raise ValueError("memory_usage_factor must be in (0, 1]")
