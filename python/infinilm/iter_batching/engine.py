"""IterBatchingEngine: per-request phased versions of step / output write-back.

Replaces ``LLMEngine.step`` and ``LLMEngine._update_requests`` (llm.py:143/178).
The main logic matches the original; only the handling of intermediate-chunk
requests differs:

- Intermediate chunk (``span.end < prompt_length``): the C++ side still samples
  1 token for the request, which must be dropped; advance
  ``num_computed_tokens`` / ``num_local_cached_tokens`` to the chunk end;
  commit the prefix cache at the chunk boundary (the original commits the
  whole prompt per ``get_total_length()``, which is a dirty publish for
  unfinished chunks); put the request back into the waiting queue to continue
  instead of running (otherwise the next step would schedule it as decode).
- Last chunk / decode: follow the original token append, text delta, and
  finish handling.
"""

from __future__ import annotations

import logging
from typing import List

from infinilm.llm.request import (
    FinishReason,
    InferenceRequest,
    RequestStatus,
    TokenOutput,
)

logger = logging.getLogger(__name__)


def step(self) -> tuple[bool, list[tuple]]:
    """Run one inference step (structured as llm.py:143, with span-based write-back)."""
    scheduler_output = self.scheduler.schedule()
    if scheduler_output is None:
        return False, []

    runner_output = self.model_runner.execute_model(scheduler_output)
    sampled_token_ids = runner_output.sampled_token_ids
    self.scheduler.update_from_output(runner_output)
    pending = self._update_requests(scheduler_output, sampled_token_ids)

    if not scheduler_output.scheduled_requests:
        if not runner_output.kv_connector_output or (
            not getattr(runner_output.kv_connector_output, "finished_sending", None)
            and not getattr(runner_output.kv_connector_output, "finished_recving", None)
        ):
            return False, pending

    return True, pending


def _update_requests(self, scheduler_output, sampled_token_ids) -> List[tuple]:
    """Apply sampled tokens (per-request phased version)."""
    requests: List[InferenceRequest] = scheduler_output.scheduled_requests
    spans = scheduler_output.spans
    if len(requests) != len(sampled_token_ids):
        raise RuntimeError(
            "model output count does not match the scheduled request count: "
            f"requests={len(requests)}, outputs={len(sampled_token_ids)}"
        )
    pending = []
    completed_path_requests = []
    for req, (start, end), token_ids in zip(requests, spans, sampled_token_ids):
        if req.is_aborted():
            logger.info(f"Request {req.request_id} aborted by client, skipping update")
            if not req.is_finished():
                req.mark_canceled()
            completed_path_requests.append(req)
            continue

        if end < req.get_prompt_length():
            # ---- Intermediate chunk: drop the sampled token, advance the boundary, requeue to waiting. ----
            req.num_computed_tokens = end
            req.num_local_cached_tokens = end
            self.scheduler.commit_computed_tokens(req, end)
            req.status = RequestStatus.WAITING
            self.scheduler.waiting_queue.sync_q.put(req)
            continue

        # ---- Last chunk / decode: same commit and write-back as the original. ----
        completed_path_requests.append(req)
        pre_output_computed_tokens = req.get_total_length()
        self.scheduler.commit_computed_tokens(req, pre_output_computed_tokens)
        # The last prefill chunk has also computed the whole prompt; sync the boundary fields.
        if end == req.get_prompt_length() and not req.generated_token_ids:
            req.num_computed_tokens = end
            req.num_local_cached_tokens = end

        if isinstance(token_ids, list):
            num_computed_output_tokens = max(len(token_ids) - 1, 0)
        else:
            num_computed_output_tokens = 0
            token_ids = [token_ids]

        num_appended_tokens = 0
        for token_id in token_ids:
            if req.is_finished():
                break
            req.append_generated_token_id(token_id)
            num_appended_tokens += 1
            pending_tokens = req.generated_token_ids[req._token_decode_offset :]
            delta = self.tokenizer.decode(pending_tokens)
            holds_back = bool(delta) and delta.endswith("�")

            last_committed_text = req.generated_text

            if not holds_back:
                req.generated_text = last_committed_text + delta
                req._token_decode_offset = len(req.generated_token_ids)

            is_finished = self._check_request_finished(req, token_id)

            # vLLM-style replacement character handling is primarily relevant for streaming.
            # For offline generation (no output queue), keep the fast incremental path.
            if req._output_queue is None:
                if is_finished:
                    req.mark_finished(req.finish_reason)

            else:
                if holds_back and not is_finished:
                    token_text = ""
                else:
                    if is_finished and req.finish_reason in (
                        FinishReason.EOS_TOKEN,
                        FinishReason.LENGTH,
                        FinishReason.STOP_STRING,
                    ):
                        token_text = ""
                    else:
                        token_text = req.generated_text[req._text_output_offset :]
                        if token_text:
                            req._text_output_offset = len(req.generated_text)

                    if is_finished:
                        req.mark_finished(req.finish_reason)

                output = TokenOutput(
                    request_id=req.request_id,
                    token_id=token_id,
                    token_text=token_text,
                    finished=is_finished,
                    finish_reason=req.finish_reason if is_finished else None,
                    generated_text=req.generated_text,
                )
                if req.is_aborted():
                    logger.info(
                        f"Request {req.request_id} aborted before putting token"
                    )
                    continue
                pending.append((req.output_queue.async_q, output))

        post_output_computed_tokens = pre_output_computed_tokens + min(
            num_appended_tokens,
            num_computed_output_tokens,
        )
        if post_output_computed_tokens > pre_output_computed_tokens:
            self.scheduler.commit_computed_tokens(req, post_output_computed_tokens)

    self.scheduler.complete_requests(completed_path_requests)
    return pending
