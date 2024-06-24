from typing import Optional

import torch
import torch.nn as nn
from vllm.model_executor.layers.sampler import (
    _apply_min_tokens_penalty, _apply_penalties, _apply_top_k_top_p,
    _apply_min_p, _sample, _get_logprobs, _build_sampler_output
)
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sequence import SamplerOutput

MASK_TOP_TH = 0.9
MASK_MAX_TIMES = 3


class ErrorSampler(nn.Module):
    def forward(
            self, logits: torch.Tensor, sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        assert logits is not None
        _, vocab_size = logits.shape

        # Apply min_tokens penalty which sets stop tokens to -inf if min_tokens
        # have not been generated yet
        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Prepare sampling tensors with pinned memory to avoid blocking.
        (sampling_tensors, do_penalties, do_top_p_top_k,
         do_min_p) = SamplingTensors.from_sampling_metadata(
            sampling_metadata, vocab_size, logits.device, logits.dtype)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties)

        # Apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits.div_(sampling_tensors.temperatures.unsqueeze_(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # ------ My custom code
        # collect seq_inds
        seq_inds = []
        for seq_group in sampling_metadata.seq_groups:
            seq_ids_, _ = seq_group
            seq_inds += seq_ids_
            # collect perturbed counters
        perturbed = []
        for i in seq_inds:
            if not hasattr(sampling_metadata.seq_data[i], 'perturbed'):
                sampling_metadata.seq_data[i].perturbed = 0
            perturbed.append(sampling_metadata.seq_data[i].perturbed)
        # help computation, cast to tensor
        seq_inds = torch.LongTensor(seq_inds).to(logits.device)
        perturbed = torch.LongTensor(perturbed).to(logits.device)
        # mask which: probability threshold, and perturbed counter
        top2 = logprobs.exp().topk(2, dim=1, sorted=True)
        should_mask_top = torch.bitwise_and(
            perturbed < MASK_MAX_TIMES, top2.values[:, 0] - top2.values[:, 1] < MASK_TOP_TH
        )
        if should_mask_top.sum() > 0:
            # actual perturb
            logits[should_mask_top, top2.indices[should_mask_top, 0]] = -float("inf")
            # re-compute softmax
            probs[should_mask_top] = torch.softmax(logits[should_mask_top], dim=-1, dtype=torch.float)
            logprobs[should_mask_top] = torch.log_softmax(logits[should_mask_top], dim=-1, dtype=torch.float)
            for i in seq_inds[should_mask_top]:  # allow at most 3 perturbations
                sampling_metadata.seq_data[int(i)].perturbed += 1
        # ------ My custom code

        # Sample the next tokens.
        sample_results = _sample(probs, logprobs, sampling_metadata,
                                 sampling_tensors)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_sampler_output(sample_results, sampling_metadata,
                                     prompt_logprobs, sample_logprobs)
