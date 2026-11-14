# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""OPSD loss ported verbatim from THU-BPM/RLCSD at a6f8afa7aa7fad82eca12984a3071840a6af4d50.

Source: third_party/verl/verl/trainer/ppo/core_algos.py (MIT).
See RLCSD_LICENSE.txt. Registration decorator omitted; formulas unchanged.
"""
import torch
import torch.nn.functional as F

def _sd_token_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1)

def _sd_cfg(config, key, default):
    """Read algorithm param from config.policy_loss."""
    if config is None:
        return default
    pl = getattr(config, 'policy_loss', None)
    if pl is None:
        return default
    if hasattr(pl, 'get'):
        value = pl.get(key, default)
    else:
        value = getattr(pl, key, default)
    return default if value is None else value

def _sd_teacher_metrics(teacher_lp, log_prob, response_mask, teacher_entropy=None):
    """Common teacher metrics for all self-distillation losses."""
    metrics = {}
    if teacher_lp is not None:
        metrics["teacher_gap"] = _sd_token_mean((teacher_lp - log_prob).abs(), response_mask).item()
        metrics["teacher_avg_logprob"] = _sd_token_mean(teacher_lp, response_mask).item()
        metrics["student_avg_logprob"] = _sd_token_mean(log_prob, response_mask).item()
    if teacher_entropy is not None:
        metrics["teacher_entropy"] = _sd_token_mean(teacher_entropy, response_mask).item()
    return metrics

def _sd_bool_cfg(config, key, default):
    return bool(_sd_cfg(config, key, default))

def _sd_int_cfg(config, key, default):
    return int(_sd_cfg(config, key, default))

def _sd_prepare_distill_log_probs(
    student_all_log_probs,
    teacher_all_log_probs,
    top_k=0,
    add_tail=True,
    top_k_source="student"):
    """Prepare aligned student/teacher log-probs for full-vocab or top-k distillation."""
    assert student_all_log_probs is not None and teacher_all_log_probs is not None, (
        "full_logit_distill=True requires student_all_log_probs and teacher_all_log_probs."
    )

    if top_k <= 0:
        return student_all_log_probs, teacher_all_log_probs

    top_k = min(top_k, student_all_log_probs.shape[-1])
    source_log_probs = student_all_log_probs if top_k_source == "student" else teacher_all_log_probs
    _, topk_indices = torch.topk(source_log_probs.detach(), k=top_k, dim=-1)
    student_topk = torch.gather(student_all_log_probs, dim=-1, index=topk_indices)
    teacher_topk = torch.gather(teacher_all_log_probs, dim=-1, index=topk_indices)

    if add_tail:
        def _tail_log_prob(log_probs):
            log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_mass = torch.clamp(log_mass, max=-1e-7)
            return torch.log(-torch.expm1(log_mass))

        student_tail = _tail_log_prob(student_topk)
        teacher_tail = _tail_log_prob(teacher_topk)
        student_topk = torch.cat([student_topk, student_tail], dim=-1)
        teacher_topk = torch.cat([teacher_topk, teacher_tail], dim=-1)
    else:
        student_topk = student_topk - torch.logsumexp(student_topk, dim=-1, keepdim=True)
        teacher_topk = teacher_topk - torch.logsumexp(teacher_topk, dim=-1, keepdim=True)

    return student_topk, teacher_topk

def _sd_prepare_sparse_topk_log_probs(student_topk_log_probs, teacher_topk_log_probs, add_tail=True):
    """Finalize aligned sparse top-k log-probs by adding a tail bucket or renormalizing."""
    assert student_topk_log_probs is not None and teacher_topk_log_probs is not None, (
        "Sparse top-k distillation requires student_topk_log_probs and teacher_topk_log_probs."
    )

    if add_tail:
        def _tail_log_prob(log_probs):
            log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_mass = torch.clamp(log_mass, max=-1e-7)
            return torch.log(-torch.expm1(log_mass))

        student_tail = _tail_log_prob(student_topk_log_probs)
        teacher_tail = _tail_log_prob(teacher_topk_log_probs)
        student_topk_log_probs = torch.cat([student_topk_log_probs, student_tail], dim=-1)
        teacher_topk_log_probs = torch.cat([teacher_topk_log_probs, teacher_tail], dim=-1)
    else:
        student_topk_log_probs = student_topk_log_probs - torch.logsumexp(
            student_topk_log_probs, dim=-1, keepdim=True
        )
        teacher_topk_log_probs = teacher_topk_log_probs - torch.logsumexp(
            teacher_topk_log_probs, dim=-1, keepdim=True
        )

    return student_topk_log_probs, teacher_topk_log_probs

def _sd_generalized_jsd(student_log_probs, teacher_log_probs, mix, vocab_clip=None):
    """Generalized JSD / KL over log-prob distributions.

    If vocab_clip is a positive float, the per-(token, vocab) divergence
    contribution is clamped at vocab_clip BEFORE summing over the vocab
    dimension. This matches the official OPSD reference implementation, where
    `F.kl_div(..., reduction="none")` (shape [B, S, V]) is clipped per element
    and then summed across both vocab and tokens. With vocab_clip=None (the
    default), behavior is unchanged: KL contributions are summed over the
    vocab dimension first and any clipping is left to the caller.
    """
    apply_vocab_clip = vocab_clip is not None and vocab_clip > 0

    if mix == 0.0:
        per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        if apply_vocab_clip:
            per_vocab = per_vocab.clamp(max=vocab_clip)
        return per_vocab.sum(-1)
    if mix == 1.0:
        per_vocab = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        if apply_vocab_clip:
            per_vocab = per_vocab.clamp(max=vocab_clip)
        return per_vocab.sum(-1)

    mix_tensor = torch.tensor(mix, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log1p(-mix_tensor),
                teacher_log_probs + torch.log(mix_tensor),
            ]
        ),
        dim=0)
    if apply_vocab_clip:
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        per_vocab = torch.lerp(kl_student, kl_teacher, mix_tensor)
        return per_vocab.clamp(max=vocab_clip).sum(-1)
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(-1)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(-1)
    return torch.lerp(kl_student, kl_teacher, mix_tensor)

def compute_policy_loss_opsd(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token_mean", config=None,
                             rollout_is_weights=None, **kwargs):
    """OPSD: Forward KL distillation (teacher -> student)."""
    teacher_lp = kwargs.get("teacher_log_probs")
    assert teacher_lp is not None, (
        "OPSD requires teacher_log_probs but got None. "
        "Ensure teacher_mode=fixed and use_kl_loss=True so ref_log_prob is computed."
    )
    beta = float(_sd_cfg(config, "beta", 0.0))
    jsd_token_clip = float(_sd_cfg(config, "jsd_token_clip", 0.05))
    full_logit_distill = _sd_bool_cfg(config, "full_logit_distill", True)
    top_k_distill = _sd_int_cfg(config, "top_k_distill", 0)
    distill_add_tail = _sd_bool_cfg(config, "distill_add_tail", True)

    if full_logit_distill:
        student_topk = kwargs.get("student_topk_log_probs")
        teacher_topk = kwargs.get("teacher_topk_log_probs")
        if top_k_distill > 0 and student_topk is not None and teacher_topk is not None:
            student_distill, teacher_distill = _sd_prepare_sparse_topk_log_probs(
                student_topk_log_probs=student_topk,
                teacher_topk_log_probs=teacher_topk,
                add_tail=distill_add_tail)
        else:
            student_all = kwargs.get("student_all_log_probs")
            teacher_all = kwargs.get("teacher_all_log_probs")
            student_distill, teacher_distill = _sd_prepare_distill_log_probs(
                student_all_log_probs=student_all,
                teacher_all_log_probs=teacher_all,
                top_k=top_k_distill,
                add_tail=distill_add_tail,
                top_k_source="teacher")
        # Match official OPSD: clip per-(token, vocab) KL contribution before
        # summing over vocab. The clip can break Gibbs' inequality and produce
        # negative per-token loss values; this matches the reference behavior.
        per_token_loss = _sd_generalized_jsd(
            student_distill,
            teacher_distill,
            beta,
            vocab_clip=jsd_token_clip if jsd_token_clip > 0 else None)
    else:
        per_token_loss = (teacher_lp.detach() - log_prob).pow(2)
        if jsd_token_clip > 0:
            per_token_loss = per_token_loss.clamp(max=jsd_token_clip)

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights
    loss = _sd_token_mean(per_token_loss, response_mask)
    metrics = {
        "distill_loss": loss.item(),
        "distill_beta": beta,
        "distill_topk": float(top_k_distill),
        "distill_full_logit": float(full_logit_distill)}
    metrics.update(_sd_teacher_metrics(teacher_lp, log_prob, response_mask, kwargs.get("teacher_entropy")))
    return loss, metrics
