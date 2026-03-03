# Diffusion Generation TODO

Status legend: `[ ]` pending, `[-]` in progress, `[x]` done.

1. [x] Fix tokenizer whitespace behavior and vocab compatibility.
   - [x] Support vocab marker conventions used in `merged_vocab.json` (`Ġ`/`▁`).
   - [x] Stop collapsing text boundaries in BPE encode path.
   - [x] Preserve sentence/paragraph whitespace better in streaming dataset path.

2. [x] Add tokenizer regression tests before retraining.
   - [x] Round-trip tests covering spaces/newlines.
   - [x] Compatibility tests for current merged vocab marker style.

3. [x] Fix inference path to use checkpoint-trained weights and proper tokenizer decode.

4. [-] Make inference perform true iterative diffusion refinement (not one-way mask fill).

5. [ ] Add automatic training-time sample dumps (`clean/corrupted/predicted`) every N steps.

6. [ ] Define and run longer training milestones on existing corpus:
   - [ ] 100k steps
   - [ ] 400k steps
   - [ ] 1M steps

7. [ ] Re-evaluate whether larger external corpora are still necessary.
