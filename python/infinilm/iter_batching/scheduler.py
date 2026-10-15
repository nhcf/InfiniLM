"""ChunkedScheduler: per-request phased scheduling + chunked prefill + optional mixed batching.

Differences from the original ``Scheduler`` (infinilm/llm/scheduler.py):

1. The scheduling output carries per-request compute intervals ``spans`` (the
   token range [start, end) each request computes in this step), replacing the
   batch-level ``is_prefill`` phase expression; ``is_prefill`` remains only as
   a legacy compatibility field (whether the batch contains prefill).
2. Chunked prefill: prefill requests are sliced by the token budget; one
   scheduling round computes only one chunk, and the remainder is requeued to
   waiting by the engine after advancing ``num_computed_tokens``.
3. Optional mixed batching (``config.mixed_batch``): decode requests take the
   budget first (Sarathi decode-maximal), and the remaining budget is filled
   with prefill chunks, mixed within the same batch.

KV allocation matches the original implementation: blocks for the whole prompt
are allocated on first scheduling (the admission-reservation logic
``can_accept_request`` is unchanged); chunking only decides "how much to
compute in this step" and never reallocates.
"""

from __future__ import annotations

import logging
import queue
from typing import List, Optional, Tuple

from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.scheduler import Scheduler, SchedulerOutput

from .config import IBConfig

logger = logging.getLogger(__name__)

Span = Tuple[int, int]


class IBSchedulerOutput(SchedulerOutput):
    """Extend `SchedulerOutput` with per-request compute intervals and phases."""

    def __init__(
        self,
        scheduled_requests: List[InferenceRequest],
        spans: List[Span],
        phases: List[str],
        **kwargs,
    ):
        is_prefill = "prefill" in phases
        super().__init__(scheduled_requests, is_prefill=is_prefill, **kwargs)
        # spans[i]: the token range [start, end) request i computes in this step.
        # Prefill chunk: end <= prompt_length; decode: (total-1, total).
        self.spans = spans
        # phases[i]: "prefill" | "decode".
        self.phases = phases


class ChunkedScheduler(Scheduler):
    """Per-request phased scheduler (chunked prefill / mixed batching)."""

    def __init__(self, *args, _ib_config: IBConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self._ib_config = _ib_config
        # The chunk budget follows the iter_batching config (the original
        # default is the whole context, i.e. no slicing).
        self.max_num_batched_tokens = _ib_config.max_batched_tokens

    # ------------------------------------------------------------------ #
    #                       First prefill scheduling                      #
    # ------------------------------------------------------------------ #
    def _first_touch_prefill(self, req: InferenceRequest, extra_blocks: int):
        """First scheduling of a prefill request: prefix hit, admission, whole-prompt allocation.

        Mirrors the ``num_computed_tokens == 0`` branch of the original
        ``Scheduler.schedule()``. Returns (ok, extra_blocks); ok=False means the
        request must be deferred (the caller requeues it). Unsupported cases
        (connector async KV) follow the original logic.
        """
        if self.enable_prefix_caching:
            (
                cached_block_table,
                num_local_computed_tokens,
            ) = self.cache_manager.get_computed_blocks(
                req.block_hashes, req.get_prompt_length() - 1
            )
        else:
            cached_block_table = []
            num_local_computed_tokens = 0
        if self.connector is not None:
            ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                req, num_local_computed_tokens
            )
            num_external_computed_tokens = ext_tokens
        else:
            load_kv_async = False
            num_external_computed_tokens = 0

        available_cached_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_computed_tokens = min(
            available_cached_tokens,
            max(req.get_prompt_length() - 1, 0),
        )
        num_new_tokens = req.get_prompt_length() - num_computed_tokens

        if not self.can_accept_request(req, num_local_computed_tokens, extra_blocks):
            logger.warning(
                "Insufficient KV cache blocks for request %s, deferring.",
                req.request_id,
            )
            if num_local_computed_tokens > 0:
                self.cache_manager.free_blocks(cached_block_table)
            return False, extra_blocks

        allocation = self.cache_manager.allocate_slots(
            num_new_tokens,
            num_computed_tokens=num_computed_tokens,
            cached_block_table=cached_block_table,
        )
        if allocation is None:
            logger.warning(
                "Failed to allocate KV cache blocks for request: %s",
                req.request_id,
            )
            if num_local_computed_tokens > 0:
                self.cache_manager.free_blocks(cached_block_table)
            return False, extra_blocks
        req_blocks, slot_mapping = allocation

        req.block_table = req_blocks
        req.slot_mapping = slot_mapping
        req.num_blocks = len(req_blocks)
        req.num_local_cached_tokens = (
            num_local_computed_tokens if load_kv_async else num_computed_tokens
        )
        req.num_cache_indexed_blocks = len(cached_block_table)
        req.num_computed_tokens = num_computed_tokens

        if self.connector is not None:
            self.connector.update_state_after_alloc(
                req, req.block_table, num_external_computed_tokens, self.block_size
            )
        if load_kv_async:
            req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            self.remote_kv_requests[req.request_id] = req
            self.pending_kv_decode_blocks += (
                req.sampling_params.max_tokens + self.block_size - 1
            ) // self.block_size
            return (
                None,
                extra_blocks,
            )  # Special state: neither scheduled nor requeued (registered for remote KV).

        return True, extra_blocks + self._get_prefill_extra_blocks(req)

    # ------------------------------------------------------------------ #
    #                          Decode scheduling                          #
    # ------------------------------------------------------------------ #
    def _schedule_decode(self, scheduled, spans, phases) -> int:
        """Schedule 1 decode token for each request in the running queue; return the token count."""
        n = 0
        while len(scheduled) < self.max_batch_size:
            try:
                req = self.running_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            if req.is_finished():
                self.complete_requests([req])
                continue
            req.block_table, new_slot = self.cache_manager.append_slot(
                req.block_table, req.get_total_length()
            )
            req.slot_mapping = [new_slot]
            req.num_blocks = len(req.block_table)
            req.num_local_cached_tokens = req.get_total_length() - 1
            total = req.get_total_length()
            scheduled.append(req)
            spans.append((total - 1, total))
            phases.append("decode")
            n += 1
        return n

    def _promote_remote_kv(self, scheduled, spans, phases):
        """Promote connector remote-KV requests (same as the original logic; a no-op without a connector)."""
        if self.connector is None or not self.remote_kv_requests:
            return
        for req_id in list(self.remote_kv_requests.keys()):
            req = self.remote_kv_requests[req_id]
            if req.is_finished():
                self.complete_requests([req])
                continue
            if req_id in self.failed_receiving_kv_req_ids:
                logger.warning(
                    f"Request {req_id[:8]}... failed receiving KV, re-queuing for prefill."
                )
                self.update_waiting_for_remote_kv(req)
                req.status = RequestStatus.WAITING
                self.waiting_queue.sync_q.put(req)
            elif req_id in self.finished_receiving_kv_req_ids:
                if len(scheduled) < self.max_batch_size:
                    logger.info(
                        f"Request {req_id[:8]}... finished receiving KV, scheduling for decode."
                    )
                    self.update_waiting_for_remote_kv(req)
                    req.status = RequestStatus.RUNNING
                    total = req.get_total_length()
                    scheduled.append(req)
                    spans.append((total - 1, total))
                    phases.append("decode")
                else:
                    break

    # ------------------------------------------------------------------ #
    #                            Main entry point                         #
    # ------------------------------------------------------------------ #
    def schedule(self) -> Optional[SchedulerOutput]:
        cfg = self._ib_config
        budget = self.max_num_batched_tokens
        scheduled: List[InferenceRequest] = []
        spans: List[Span] = []
        phases: List[str] = []
        deferred: List[InferenceRequest] = []
        tokens = 0
        extra_blocks = 0

        # Mixed batching: decode takes the budget first (decode-maximal).
        # Without mixing, decode runs only when no prefill can be scheduled
        # (same as the original behavior).
        if cfg.mixed_batch:
            tokens += self._schedule_decode(scheduled, spans, phases)
            self._promote_remote_kv(scheduled, spans, phases)

        # A mixed step (decode already in the batch) may optionally cap prefill
        # tokens (a knob trading ITL stability against prefill speed; unlimited
        # measured best for throughput and TTFT tail at c128, default 0 =
        # unlimited).
        mixed_with_decode = cfg.mixed_batch and tokens > 0
        prefill_cap = (
            cfg.mixed_step_tokens
            if mixed_with_decode and cfg.mixed_step_tokens > 0
            else budget
        )
        prefill_tokens = 0

        # Prefill: fill the remaining budget with chunks.
        while (
            len(scheduled) < self.max_batch_size
            and tokens < budget
            and prefill_tokens < prefill_cap
        ):
            try:
                req = self.waiting_queue.sync_q.get_nowait()
            except queue.Empty:
                break
            if req.is_finished():
                self.complete_requests([req])
                continue

            if req.num_computed_tokens == 0:
                if (
                    req.has_multimodal_inputs or not cfg.chunked_prefill
                ) and self._exceeds_token_budget(
                    tokens, req.get_prompt_length(), len(scheduled)
                ):
                    # The budget check for whole-segment scheduling must run
                    # before block allocation (same as the original Scheduler):
                    # `_first_touch_prefill` allocates blocks for the whole
                    # prompt, so deferring afterwards would carry the blocks
                    # back to the waiting queue on the request, and the retry
                    # would allocate again, leaking blocks.
                    deferred.append(req)
                    break
                ok, extra_blocks = self._first_touch_prefill(req, extra_blocks)
                if ok is None:
                    continue  # Moved to remote-KV waiting.
                if not ok:
                    deferred.append(req)
                    break

            start = req.num_computed_tokens
            remaining = req.get_prompt_length() - start
            if req.has_multimodal_inputs or not cfg.chunked_prefill:
                # Multimodal / chunking disabled: schedule the whole segment,
                # keeping the original budget check (single-request exemption).
                if self._exceeds_token_budget(tokens, remaining, len(scheduled)):
                    deferred.append(req)
                    break
                chunk = remaining
            else:
                chunk = min(remaining, budget - tokens, prefill_cap - prefill_tokens)
            if chunk <= 0:
                deferred.append(req)
                break

            end = start + chunk
            # KV write positions for this chunk (first scheduling already
            # allocated blocks for the whole prompt; only rebuild the slot
            # mapping for the [start, end) range here).
            req.slot_mapping = self.cache_manager.update_blocks_slot(
                req.block_table, start, end
            )
            req.num_local_cached_tokens = start
            scheduled.append(req)
            spans.append((start, end))
            phases.append("prefill")
            tokens += chunk
            prefill_tokens += chunk
            req.status = RequestStatus.RUNNING

        if deferred:
            for req in deferred:
                self.waiting_queue.sync_q.put(req)

        # Without mixing: build a decode batch only when no prefill was
        # scheduled (same as the original behavior).
        if not scheduled and not cfg.mixed_batch:
            tokens += self._schedule_decode(scheduled, spans, phases)
            self._promote_remote_kv(scheduled, spans, phases)

        if scheduled:
            out = IBSchedulerOutput(
                scheduled_requests=scheduled,
                spans=spans,
                phases=phases,
                speculative_cache_ops=self.speculative_cache_ops,
            )
            if self.connector is not None:
                out.kv_connector_metadata = self.connector.build_connector_meta()
            return out

        if self.connector is not None:
            out = IBSchedulerOutput(
                scheduled_requests=[],
                spans=[],
                phases=[],
                speculative_cache_ops=self.speculative_cache_ops,
            )
            out.kv_connector_metadata = self.connector.build_connector_meta()
            return out

        return None
