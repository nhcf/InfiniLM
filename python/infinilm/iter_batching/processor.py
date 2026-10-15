"""Unified per-request batch input building.

Replaces the batch-level ``is_prefill`` branch of
``BasicLLMProcessor._build_model_input_from_batch_scheduler_output``
(processors/basic_llm_processor.py:161): instead of fetching tokens by
"whole-batch prefill / whole-batch decode", tokens are packed by the
per-request intervals ``spans`` ([start, end)) given by the scheduler. Prefill
chunks and decode requests (``end - start == 1``) share an identical format
when mixed in one batch.

The output tensor fields are exactly the same as the original (transparent to
the C++ side):
input_ids / position_ids / past_kv_lengths / total_kv_lengths /
input_offsets / cu_seqlens / block_tables / slot_mapping /
temperature / top_k / top_p
"""

from __future__ import annotations

from infinilm.llm.scheduler import SchedulerOutput


def build_model_inputs(
    self, scheduler_output: SchedulerOutput, temperature, top_p, top_k
) -> dict:
    """Build mixed-batch inputs from per-request spans (bound as a processor method)."""
    import infinicore

    if not scheduler_output.scheduled_requests:
        raise RuntimeError("build_model_inputs called with empty scheduled_requests")

    spans = scheduler_output.spans
    tokens = []
    seq_lens = []
    seq_offsets = [0]
    block_tables = []
    slot_mapping = []
    cached_lens = []
    position_ids = []
    cu_seqlens = [0]

    max_block_table_len = max(
        len(req.block_table) for req in scheduler_output.scheduled_requests
    )
    current_offset = 0

    for req, (start, end) in zip(scheduler_output.scheduled_requests, spans):
        if end - start > 1 or end <= req.get_prompt_length():
            # Prefill (including chunks): take the [start, end) segment of the prompt.
            seg = req.prompt_token_ids[start:end]
            tokens.extend(seg)
            position_ids.extend(range(start, end))
        else:
            # Decode: the last generated token (span is (total-1, total)).
            last_token = (
                req.generated_token_ids[-1]
                if req.generated_token_ids
                else req.prompt_token_ids[-1]
            )
            tokens.append(last_token)
            position_ids.append(end - 1)

        seq_lens.append(end)  # Total visible KV length of the request (chunk end).
        current_offset += end - start
        seq_offsets.append(current_offset)
        slot_mapping.extend(req.slot_mapping)
        cached_lens.append(start)
        block_tables.append(
            req.block_table + [-1] * (max_block_table_len - len(req.block_table))
        )
        cu_seqlens.append(cu_seqlens[-1] + end)

    return {
        "input_ids": infinicore.from_list([tokens], dtype=infinicore.int64),
        "position_ids": infinicore.from_list(position_ids, dtype=infinicore.int64),
        "past_kv_lengths": infinicore.from_list(cached_lens, dtype=infinicore.int32),
        "total_kv_lengths": infinicore.from_list(seq_lens, dtype=infinicore.int32),
        "input_offsets": infinicore.from_list(seq_offsets, dtype=infinicore.int32),
        "cu_seqlens": infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
        "block_tables": infinicore.from_list(block_tables, dtype=infinicore.int32),
        "slot_mapping": infinicore.from_list(slot_mapping, dtype=infinicore.int64),
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
