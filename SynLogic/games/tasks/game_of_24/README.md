## About
This folder is for generating game of 24 questions and answers.

To generate the questions and answers, you can run the `./run.sh` command.

## Folder Structure
- `./scripts/game_of_24.py`: the core logic for generating game of 24 questions and answers.
- `./scripts/game_of_24_prompt.py`: the prompt for generating game of 24 questions and answers.
- `./scripts/game_of_24_verifier.py`: the verifier for verifying game of 24 answers.
- `./run.sh`: a script for generating game of 24 questions and answers.

## Scoring protocol

`mathador-target-v2` checks the expression against `metadata.result` in the native
verifier, shared by training rewards, validation and direct verification. It keeps
the existing answer extraction, number/operator checks and `1e-10` tolerance.
The earlier verifier overwrote the target and compared the result with itself.
Historical results using that verifier retain their original scoring identity;
the correction does not re-score them or add Mathador to existing datasets.

