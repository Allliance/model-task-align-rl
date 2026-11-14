"""Absolute outer-step schedule shared by driver, workers and checkpoint guards."""


def phase_for_step(step, block_steps=10):
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError('Alternating policy requires a positive absolute outer step')
    if block_steps != 10:
        raise ValueError('This protocol uses exactly ten outer steps per phase')
    number = (step - 1) // block_steps
    ta = number % 2 == 1
    return dict(outer_step=step, phase='ta_opd' if ta else 'rl',
                phase_number=number + 1, phase_offset=(step - 1) % block_steps + 1,
                block_steps=block_steps, teacher_coefficient=float(ta),
                outcome_coefficient=float(not ta))


def checkpoint_phase(step):
    return dict(schedule='rl10-ta_opd10-v1', completed_outer_step=step,
                completed_phase=phase_for_step(step) if step else None,
                next_phase=phase_for_step(step + 1))


def validate_checkpoint_phase(state):
    if state.get('alternating_phase') != checkpoint_phase(state['step']):
        raise ValueError('Checkpoint absolute alternating schedule differs')
    if state['optimizer_updates'] != state['step'] * 5:
        raise ValueError('Alternating checkpoint optimizer counter differs')
