"""Soft trust from on-policy advantages and clean-reference disagreement.

Only observed pool membership, response masks, UIDs, log probabilities and the
assigned-label GRPO advantages enter this calculation. Original answers and
synthetic corruption indicators are deliberately unavailable to the scorer.
"""
from collections import defaultdict
import torch


def guidance_weights(advantages, ref_log_prob, old_log_probs, response_mask,
                     uids, pools, condition, group_size=8):
    allowed=('base-kl','guide-kl','guide-adaptive-kl','guide-soft-pg')
    if condition not in allowed:
        raise ValueError('Unknown guided objective')
    mask=response_mask.detach().float()
    if advantages.shape!=mask.shape or ref_log_prob.shape!=mask.shape or old_log_probs.shape!=mask.shape:
        raise ValueError('Guidance token tensors must have identical shapes')
    lengths=mask.sum(-1)
    if not bool((lengths>0).all()):
        raise ValueError('An empty response cannot have a guidance score')
    advantage=(advantages.detach().float()*mask).sum(-1)/lengths
    # Current policy, not the untouched base: this is a local direction proxy.
    displacement=((ref_log_prob.detach().float()-old_log_probs.detach().float())*mask).sum(-1)/lengths
    if not bool(torch.isfinite(advantage).all() and torch.isfinite(displacement).all()):
        raise FloatingPointError('Non-finite guidance inputs')
    if len(uids)!=len(mask) or len(pools)!=len(mask) or any(pool not in ('small','large') for pool in pools):
        raise ValueError('Guidance requires frozen pool membership for every response')
    groups=defaultdict(list)
    for index,uid in enumerate(uids):
        groups[str(uid)].append(index)
    conflict=torch.zeros_like(advantage)
    active=0
    for indices in groups.values():
        if len(indices)!=group_size or len({pools[index] for index in indices})!=1:
            raise ValueError('Each UID must preserve its complete rollout group and pool')
        a=advantage[indices]; a=a-a.mean()
        delta=displacement[indices]; delta=delta-delta.mean()
        norm=a.norm()*delta.norm()
        if bool(a.norm()>1e-8):
            active+=1
        if bool(norm>1e-12):
            alignment=(a.dot(delta)/norm).clamp(-1,1)
            conflict[indices]=(-alignment).clamp(0,1)
    is_large=torch.tensor([pool=='large' for pool in pools],device=mask.device,dtype=torch.bool)
    conflict=conflict*is_large
    pg=torch.ones_like(advantage)
    kl=torch.ones_like(advantage)
    if condition in ('base-kl','guide-kl'):
        kl[is_large]=10.
    elif condition=='guide-adaptive-kl':
        kl=1.+9.*conflict
    else:
        pg=1.-0.9*conflict
        # Existing backend weights all terms by sample_weights. Compensate KL
        # so this arm tests only PPO attenuation, with the original KL retained.
        kl=pg.reciprocal()
    metrics={
        'guided/large_response_fraction':float(is_large.float().mean()),
        'guided/active_prompt_fraction':active/len(groups),
        'guided/conflict_mean':float(conflict.mean()),
        'guided/conflicting_response_fraction':float((conflict>0).float().mean()),
        'guided/pg_weight_mean':float(pg.mean()),
        'guided/effective_kl_multiplier_mean':float((pg*kl).mean()),
        'guided/large_conflict_mean':float(conflict[is_large].mean()) if bool(is_large.any()) else 0.,
        'guided/large_effective_kl_multiplier_mean':float((pg*kl)[is_large].mean()) if bool(is_large.any()) else 1.,
    }
    return pg,kl,metrics


def prepare_guided_batch(data,config):
    condition=config.guided.condition
    pools=[extra.get('pool') for extra in data.non_tensor_batch['extra_info']]
    pg,kl,metrics=guidance_weights(data.batch['advantages'],data.batch['ref_log_prob'],
        data.batch['old_log_probs'],data.batch['response_mask'],data.non_tensor_batch['uid'],pools,
        condition,group_size=config.actor_rollout_ref.rollout.n)
    if 'sample_weights' in data.batch and not bool((data.batch['sample_weights']==1).all()):
        raise ValueError('This pilot may not silently combine multiple PG weighting schemes')
    data.batch['kl_weights']=kl
    if condition=='guide-soft-pg':
        data.batch['sample_weights']=pg
    return metrics
