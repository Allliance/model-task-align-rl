# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # Padded tensor width can exceed native context; only valid tokens
            # and their positions constrain a packed/padded causal sequence.
            unwrapped=self.actor_module
            while not hasattr(unwrapped,'config'):
                unwrapped=unwrapped.module if hasattr(unwrapped,'module') else unwrapped.model
            model_config=unwrapped.config
            native_limit=model_config.max_position_embeddings
            valid=micro_batch['attention_mask'].bool()
            assert (valid.sum(-1)<=native_limit).all(), 'Valid sequence exceeds native context'
            if position_ids.dim()==2:
                assert (position_ids[valid]<native_limit).all(), 'Valid position exceeds native context'
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    if 'opsd_teacher_ids' in micro_batch:
                        self._opsd_student_lp=self._opsd_probabilities(logits_rmpad,micro_batch,indices)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = "opsd_teacher_ids" not in micro_batch
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    if 'opsd_teacher_ids' in micro_batch:
                        self._opsd_student_lp=self._opsd_probabilities(logits,micro_batch)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"],
                                                     inplace_backward="opsd_teacher_ids" not in micro_batch)
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _opsd_probabilities(self,logits,micro_batch,indices=None):
        assert not self.use_ulysses_sp and not self.use_fused_kernels
        response_mask=micro_batch['attention_mask'][:,-micro_batch['responses'].shape[1]:].bool()
        b,r=response_mask.shape;s=micro_batch['input_ids'].shape[1]
        positions=(torch.arange(b,device=logits.device)[:,None]*s+torch.arange(s-r-1,s-1,device=logits.device)[None,:]).flatten()
        valid=response_mask.flatten();positions=positions[valid]
        if indices is not None:
            packed=torch.searchsorted(indices,positions)
            assert torch.equal(indices[packed],positions),'Response prediction positions absent from packed sequence'
            selected=logits[packed]
        else:selected=logits.reshape(-1,logits.shape[-1])[positions]
        ids=micro_batch['opsd_teacher_ids'].reshape(-1,micro_batch['opsd_teacher_ids'].shape[-1])[valid].long()
        z=selected.float();normalizer=torch.logsumexp(z,-1,keepdim=True);lp=z.gather(-1,ids)-normalizer
        labels=micro_batch['responses'].flatten()[valid,None]
        chosen=z.gather(-1,labels).squeeze(-1)-normalizer.squeeze(-1)
        self._opsd_alignment_lp=chosen.detach()

        output=lp.new_zeros((b*r,ids.shape[-1])).index_copy(0,valid.nonzero().flatten(),lp)
        return output.reshape(b,r,-1)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        # Shared-current retrieval keeps rollout, teacher and student phases on
        # the same device. Release only inactive allocator blocks; leave the
        # original scoring chunks, tensors and returned values unchanged.
        reclaim_scoring = self.config.get('shared_current_opsd', False) and not use_dynamic_bsz
        profile_once = reclaim_scoring and not getattr(self, '_scoring_memory_profiled', False)
        import time
        call_started = time.time()
        completed_rows = 0
        for micro_index, micro_batch in enumerate(micro_batches, 1):
            if reclaim_scoring:
                torch.cuda.empty_cache()
            if profile_once:
                from src.scoring_memory import record_scoring_memory
                record_scoring_memory('before_forward', micro_index - 1, micro_batch)
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            if profile_once:
                record_scoring_memory('after_forward', micro_index - 1, micro_batch)
            log_probs_lst.append(log_probs)
            completed_rows += len(micro_batch['responses'])
            if os.environ.get('SYNLOGIC_RUN_ROOT') and os.environ.get('SYNLOGIC_ALLOCATION_ID') and os.environ.get('SYNLOGIC_ATTEMPT'):
                # Publish only completed forward work, with no text or token IDs.
                import json
                from pathlib import Path
                valid_tokens = int(micro_batch['attention_mask'].sum().item())
                role = 'actor' if self.actor_optimizer is not None else 'reference'
                target = Path(os.environ['SYNLOGIC_RUN_ROOT']) / f'inference-progress-{role}-rank{torch.distributed.get_rank()}.json'
                progress = dict(epoch=time.time(), call_started=call_started, allocation_id=os.environ['SYNLOGIC_ALLOCATION_ID'],
                                attempt=os.environ['SYNLOGIC_ATTEMPT'], role=role, rank=torch.distributed.get_rank(),
                                completed_micro_batches=micro_index, total_micro_batches=len(micro_batches),
                                completed_rows=completed_rows, total_rows=len(batch), last_valid_tokens=valid_tokens)
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps(progress)+'\n')
                temporary.replace(target)
            if calculate_entropy:
                entropy_lst.append(entropy)

        if reclaim_scoring:
            torch.cuda.empty_cache()
        self._scoring_memory_profiled = True
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        if torch.cuda.is_available():
            # Release completed rollout/logprob workspaces before training.
            torch.cuda.empty_cache()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        shared_opsd=self.config.get('shared_current_opsd',False)
        oracle_opsd=self.config.get('oracle_opsd',False) or shared_opsd
        if oracle_opsd:
            assert not self.config.use_kl_loss and self.config.entropy_coeff==0
            if shared_opsd:
                assert data.meta_info['shared_current_opsd'] and hasattr(self,'shared_teacher_callback')
                select_keys += ['opsd_privileged_'+key for key in ['prompts','input_ids','attention_mask','position_ids']]
                select_keys += [key for key in ['opsd_capture_targets','opsd_snapshot_row_index','opsd_teacher_policy_version'] if key in data.batch]
                policy_outer_step=int(data.meta_info['opsd_policy_outer_step'])
            else:
                select_keys += ['opsd_teacher_ids','opsd_teacher_lp','teacher_log_probs']
        separate_opd = self.config.get("opd_loss_mode") == "importance_sampling"
        if separate_opd:
            if (data.meta_info.get("opd_loss_mode") != "importance_sampling"
                    or data.meta_info.get("opd_teacher_coefficient") != self.config.opd_teacher_coefficient
                    or data.meta_info.get("opd_outcome_coefficient") != self.config.opd_outcome_coefficient
                    or "global_minibatch_response_tokens" not in data.batch):
                raise ValueError("Actor OPD objective differs from the checked trainer objective")
        elif data.meta_info.get("opd_loss_mode") == "importance_sampling":
            raise ValueError("Trainer requested separate OPD importance sampling but actor mode differs")
        if separate_opd:
            select_keys += ["teacher_advantages", "outcome_advantages"]
        if "opd_token_mask" in data.batch:
            select_keys += ["opd_token_mask", "global_opd_response_tokens"]
        if "legacy_global_minibatch_rows" in data.batch:
            if separate_opd or "global_minibatch_response_tokens" in data.batch:
                raise ValueError("Legacy and global token normalization are mutually exclusive")
            if "sample_weights" not in data.batch:
                raise ValueError("Legacy microbatch preservation requires original loss weights")
            if self.config.loss_agg_mode != "token-mean":
                raise ValueError("Legacy microbatch preservation requires token-mean")
            select_keys.append("legacy_global_minibatch_rows")
        if "global_minibatch_response_tokens" in data.batch:
            select_keys.append("global_minibatch_response_tokens")
        if "sample_weights" in data.batch:
            select_keys.append("sample_weights")
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)
        if shared_opsd:
            assert self.config.ppo_epochs==1 and len(dataloader)==5,'Shared teacher version contract requires exactly five optimizer updates per outer'

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    if shared_opsd:
                        # Plan execution using both contexts; optimization still
                        # receives original student masks and global token means.
                        work=mini_batch['attention_mask'].sum(-1)+mini_batch['opsd_privileged_attention_mask'].sum(-1)
                        assert int(work.max())<=max_token_len, (f'One paired context exceeds the execution token budget: paired_max={int(work.max())}, budget={max_token_len}, actor_budget={self.config.ppo_max_token_len_per_gpu}, sp={self.ulysses_sequence_parallel_size}, student_max={int(mini_batch["attention_mask"].sum(-1).max())}, teacher_max={int(mini_batch["opsd_privileged_attention_mask"].sum(-1).max())}, student_width={mini_batch["attention_mask"].shape[-1]}, teacher_width={mini_batch["opsd_privileged_attention_mask"].shape[-1]}')
                        planner=mini_batch.clone()
                        planner['attention_mask']=torch.arange(int(work.max()),device=work.device)[None,:]<work[:,None]
                        minimum=None
                        while True:
                            _,parts=rearrange_micro_batches(planner,max_token_len=max_token_len,min_num_micro_batch=minimum)
                            overflow=torch.tensor(int(any(work[rows].sum()>max_token_len for rows in parts)),device=get_device_id())
                            torch.distributed.all_reduce(overflow,op=torch.distributed.ReduceOp.MAX)
                            if not overflow.item():break
                            minimum=len(parts)+1
                            assert minimum<=len(mini_batch),'Unable to prepare bounded shared-context microbatches'
                        micro_batches=[mini_batch[rows] for rows in parts]
                    else:
                        micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                opd_minibatch_loss = 0.0

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = (data["outcome_advantages"] * self.config.opd_outcome_coefficient
                                  if separate_opd else data["advantages"])
                    sample_weights = data.get("sample_weights")

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    if shared_opsd:
                        policy_version=(policy_outer_step-1)*len(dataloader)+epoch*len(dataloader)+batch_idx
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        targets=self.shared_teacher_callback(data,temperature,policy_version)
                        if torch.cuda.is_available():
                            # The no-grad teacher's logits/workspaces are dead;
                            # leave physical headroom for external BLAS kernels.
                            torch.cuda.empty_cache()
                        assert all(not value.requires_grad for value in targets.values())
                        data.update(targets)
                        metrics['actor/opsd_teacher_policy_version']=float(policy_version)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    if oracle_opsd:
                        # Direct FKL has no PPO objective, even as a discarded
                        # intermediate. Preserve ordinary metric shapes only.
                        pg_loss=log_prob.sum()*0
                        pg_clipfrac=ppo_kl=pg_clipfrac_lower=log_prob.new_zeros(())
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                        sample_weights=sample_weights,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask,
                                                loss_agg_mode=loss_agg_mode, sample_weights=sample_weights)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if oracle_opsd and not getattr(self,'_opsd_alignment_checked',False):
                        difference=(self._opsd_alignment_lp-log_prob[response_mask.bool()].detach()).abs()
                        assert float(difference.mean())<.03 and float(difference.max())<1.,'OPSD actor causal probability alignment failed'
                        from pathlib import Path
                        import json,time
                        root=Path(os.environ['SYNLOGIC_RUN_ROOT']);rank=torch.distributed.get_rank()
                        (root/f'opsd-gpu-alignment-student-rank{rank}.json').write_text(json.dumps({'passed':True,'positions':len(difference),'mean_abs_difference':float(difference.mean()),'max_abs_difference':float(difference.max()),'epoch':time.time()}))
                        self._opsd_alignment_checked=True
                    if oracle_opsd:
                        from verl.trainer.ppo.opsd_loss import compute_policy_loss_opsd
                        policy_loss,opsd_metrics=compute_policy_loss_opsd(
                            old_log_prob,log_prob,advantages,response_mask,config=self.config,
                            teacher_log_probs=data['teacher_log_probs'].detach(),
                            student_topk_log_probs=self._opsd_student_lp,
                            teacher_topk_log_probs=data['opsd_teacher_lp'].detach())
                        del self._opsd_student_lp
                        metrics.update({'actor/opsd_'+key:value for key,value in opsd_metrics.items()})
                        pg_loss=policy_loss

                    if separate_opd:
                        selected_mask=response_mask * data.get("opd_token_mask",torch.ones_like(response_mask))
                        if selected_mask.any():
                            opd_loss, opd_clipfrac, _, _ = compute_policy_loss(
                                old_log_prob=old_log_prob,log_prob=log_prob,
                                advantages=data["teacher_advantages"].detach(),response_mask=selected_mask,
                                cliprange=clip_ratio,cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,sample_weights=sample_weights)
                        else:
                            opd_loss=log_prob.sum()*0
                            opd_clipfrac=log_prob.new_zeros(())
                        if "global_opd_response_tokens" in data:
                            # Outer backward weights by original tokens / global original tokens.
                            # Rescale only the teacher term to obtain its independent selected-token mean.
                            factor=(selected_mask.sum()/response_mask.sum().clamp_min(1)
                                    *data["global_minibatch_response_tokens"][0]/data["global_opd_response_tokens"][0])
                            opd_loss=opd_loss*factor
                        metrics["actor/opd_clipfrac"] = float(opd_clipfrac.detach())
                        policy_loss = policy_loss + self.config.opd_teacher_coefficient * opd_loss
                        dp_size = torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
                        opd_minibatch_loss += float(opd_loss.detach() * response_mask.sum() * dp_size
                                                    / data["global_minibatch_response_tokens"][0])
                        metrics["actor/opd_teacher_coefficient"] = self.config.opd_teacher_coefficient

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                                           loss_agg_mode=loss_agg_mode, sample_weights=sample_weights)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if "legacy_global_minibatch_rows" in data:
                        # sample_weights encode the original SP microbatch's
                        # rows/tokens ratio. Preserve that existing objective
                        # while distributing its same minibatches across DP.
                        denominator = data["legacy_global_minibatch_rows"][0]
                        if not torch.all(data["legacy_global_minibatch_rows"] == denominator):
                            raise ValueError("Mixed legacy optimizer minibatches in a microbatch")
                        dp_size = torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
                        loss = policy_loss * response_mask.sum() * dp_size / denominator
                    elif "global_minibatch_response_tokens" in data:
                        if loss_agg_mode != "token-mean":
                            raise ValueError("Global response-token normalization requires token-mean")
                        denominator = data["global_minibatch_response_tokens"][0]
                        if not torch.all(data["global_minibatch_response_tokens"] == denominator):
                            raise ValueError("Mixed optimizer minibatches in a microbatch")
                        dp_size = torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
                        loss = policy_loss * response_mask.sum() * dp_size / denominator
                    elif self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                if shared_opsd:
                    assert torch.isfinite(grad_norm),'Nonfinite shared OPSD gradient skipped an Adam update; fail closed'
                data = {"actor/grad_norm": grad_norm.detach().item()}
                if separate_opd:
                    # DataProto.concat retains rank-zero metadata. Publish the
                    # actual global loss rather than rank zero's DP-scaled share.
                    reported = torch.tensor(opd_minibatch_loss, device=get_device_id())
                    torch.distributed.all_reduce(reported)
                    reported /= torch.distributed.get_world_size()
                    data["actor/opd_loss"] = reported.item()
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
