"""Phase boundaries and checkpoint continuity for alternating outer steps."""
import copy
import pytest
from src.alternating_opd import phase_for_step, checkpoint_phase, validate_checkpoint_phase


@pytest.mark.parametrize('step,phase,offset', [
    (1, 'rl', 1), (10, 'rl', 10), (11, 'ta_opd', 1),
    (20, 'ta_opd', 10), (21, 'rl', 1), (600, 'ta_opd', 10),
])
def test_absolute_phase_boundaries(step, phase, offset):
    value = phase_for_step(step)
    assert value['phase'] == phase and value['phase_offset'] == offset
    assert value['teacher_coefficient'] + value['outcome_coefficient'] == 1
    assert value['teacher_coefficient'] * value['outcome_coefficient'] == 0


@pytest.mark.parametrize('completed', [9, 10, 19, 20, 299, 300])
def test_resume_preserves_the_next_absolute_phase(completed):
    state = dict(step=completed, optimizer_updates=completed * 5,
                 alternating_phase=checkpoint_phase(completed))
    validate_checkpoint_phase(state)
    assert state['alternating_phase']['next_phase'] == phase_for_step(completed + 1)
    wrong = copy.deepcopy(state)
    wrong['alternating_phase']['next_phase'] = phase_for_step(1)
    with pytest.raises(ValueError, match='schedule'):
        validate_checkpoint_phase(wrong)
    state['optimizer_updates'] += 1
    with pytest.raises(ValueError, match='counter'):
        validate_checkpoint_phase(state)


def test_target_has_balanced_phases():
    phases = [phase_for_step(i) for i in range(1, 601)]
    assert sum(p['phase'] == 'rl' for p in phases) == 300
    assert sum(p['phase'] == 'ta_opd' for p in phases) == 300
    assert phases[-1]['phase_number'] == 60
