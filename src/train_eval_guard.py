"""Fail-closed checks of resolved SynLogic training inputs and held-out puzzles."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata


FIELDS = {
    'buggy_tables': ['buggy_table', 'queries', 'bug_description'],
    'calcudoko': ['grid_size', 'regions'],
    'campsite': ['grid', 'row_constraints', 'col_constraints'],
    'cryptarithm': ['letter_words', 'operators'],
    'dyck_language': ['question_sequence'],
    'dyck_language_errors': ['bracket_string'],
    'dyck_language_reasoning_errors': ['dyck_sequence', 'thoughts'],
    'math_path': ['ref_expr', 'query_expr'],
    'mathador': ['numbers', 'operators', 'result'],
    'number_wall': ['grid'], 'numbrix': ['grid'],
    'object_properties': ['story', 'question', 'initial_collection'],
    'operation': ['expression', 'symbol_definitions'],
    'space_reasoning': ['expression', 'symbol_definitions'],
    'skyscraper_puzzle': ['n', 'top', 'bottom', 'left', 'right'],
    'star_placement_puzzle': ['n', 'k', 'region_grid'],
    'sudoku': ['original_sudoku'],
    'survo': ['original_matrix', 'candidate_numbers'],
    'time_sequence': ['n', 'records'],
    'web_of_lies': ['people', 'statements'],
    'word_sorting': ['words', 'front_letters', 'order'],
    'word_sorting_mistake': ['words', 'thoughts'],
    'zebra_puzzle': ['question'],
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def normalized(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value)).strip()


def signatures(row):
    task = row['data_source'].removeprefix('val/')
    extra = row['extra_info']
    game = json.loads(extra.get('original_game_data_str') or extra['game_data_str'])
    question = game['question'].split('<answer> answer here</answer>.', 1)[-1]
    result = {('prompt', normalized(canonical(row['prompt']))),
              ('original_question', normalized(question))}
    metadata = game.get('metadata') or {}
    fields = FIELDS.get(task)
    if fields and all(k in metadata for k in fields):
        content = {key: metadata[key] for key in fields}
        if task == 'web_of_lies':
            # Truth assignments and generated solutions are not puzzle inputs.
            content['people'] = [{key: person[key] for key in ('id', 'name', 'location') if key in person}
                                 for person in content['people']]
        if task == 'star_placement_puzzle':
            names = {}
            content['region_grid'] = [[names.setdefault(v, len(names)) for v in line]
                                      for line in content['region_grid']]
        result.add(('puzzle', task, canonical(content)))
    return result


def check_rows(training, validation, excluded_source_indices=(), additional_questions=()):
    heldout = set().union(*(signatures(row) for row in validation))
    heldout.update(('original_question', normalized(question)) for question in additional_questions)
    forbidden = set(excluded_source_indices)
    collisions = []
    for row in training:
        index = int(row['extra_info']['index'])
        matched = signatures(row) & heldout
        if index in forbidden or matched:
            collisions.append({'source_index': index, 'task': row['data_source'],
                               'signature_types': sorted({s[0] for s in matched})})
    if collisions:
        raise ValueError(f'Training contains {len(collisions)} held-out or previously excluded inputs; '
                         f'first collisions: {collisions[:3]}')
    return {'training_rows': len(training), 'validation_rows': len(validation),
            'known_input_overlap_count': 0,
            'limitation': 'Fingerprint nonmatches do not prove semantic uniqueness.'}


def check_resolved_inputs(train_files, validation_files, expected_training_sha256,
                          exclusions_file, additional_questions_file=None):
    import pyarrow.parquet as pq
    paths = [train_files] if isinstance(train_files, str) else list(train_files)
    validation_paths = [validation_files] if isinstance(validation_files, str) else list(validation_files)
    if len(paths) != 1:
        raise ValueError('This frozen protocol expects exactly one training parquet')
    actual = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
    if actual != expected_training_sha256:
        raise ValueError('Resolved training parquet differs from the approved frozen fingerprint')
    exclusions = json.loads(Path(exclusions_file).read_text())['excluded_source_indices']
    additional = json.loads(Path(additional_questions_file).read_text()) if additional_questions_file else []
    training = pq.read_table(paths[0]).to_pylist()
    validation = [row for path in validation_paths for row in pq.read_table(path).to_pylist()]
    return {**check_rows(training, validation, exclusions, additional), 'training_sha256': actual}
