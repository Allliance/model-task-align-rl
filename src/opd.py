"""Immediate sampled reverse-KL feedback on student-generated response tokens."""
import math

import torch


def combine_advantages(outcome, teacher_log_probs, old_log_probs, response_mask,
                       outcome_coefficient, teacher_coefficient):
    if any(t.shape != outcome.shape for t in (teacher_log_probs, old_log_probs, response_mask)):
        raise ValueError('OPD tensors must have identical response-token shapes')
    if not all(math.isfinite(float(v)) and v >= 0 for v in (outcome_coefficient, teacher_coefficient)):
        raise ValueError('Invalid OPD coefficient')
    mask = response_mask.bool()
    if not mask.any():
        raise ValueError('OPD batch has no response tokens')
    for value in (outcome, teacher_log_probs, old_log_probs):
        if not torch.isfinite(value[mask]).all():
            raise FloatingPointError('Non-finite active OPD input')
    # where, rather than multiplication, also excludes NaN/Inf padding.
    dense = torch.where(mask, teacher_log_probs.detach() - old_log_probs.detach(), 0.)
    advantage = torch.where(mask, outcome_coefficient * outcome.detach()
                            + teacher_coefficient * dense, 0.)
    if not torch.isfinite(advantage).all():
        raise FloatingPointError('Non-finite combined OPD advantage')
    metrics = {'opd/teacher_coefficient': float(teacher_coefficient),
               'opd/outcome_coefficient': float(outcome_coefficient),
               'opd/teacher_log_ratio_mean': dense[mask].mean().item(),
               'opd/teacher_log_ratio_abs_mean': dense[mask].abs().mean().item(),
               'opd/outcome_advantage_abs_mean': outcome[mask].abs().mean().item(),
               'opd/combined_advantage_abs_mean': advantage[mask].abs().mean().item()}
    return advantage.detach(), dense.detach(), metrics


def gradient_matched_coefficient(outcome_norms, teacher_norms):
    if len(outcome_norms) != 4 or len(teacher_norms) != 4:
        raise ValueError('Calibration requires four optimizer minibatches per component')
    values = [float(x) for x in outcome_norms + teacher_norms]
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise FloatingPointError('Invalid calibration gradient norm')
    left, right = sum(outcome_norms) / 4, sum(teacher_norms) / 4
    if left <= 0 or right <= 0:
        raise ValueError('Calibration component has no gradient signal')
    return left / right
