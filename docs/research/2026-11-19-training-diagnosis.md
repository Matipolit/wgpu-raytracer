# Why the v4 networks plateaued, and what changes it

Written 2026-11-19. The v4 run (scripts/overnight_v4.sh) retrained the temporal
denoisers for +0.3 to +0.45 dB and trained the first fast joint network; both
plateaued within an hour of a six-hour run. A fresh reviewer (a subagent with no
context, given the code, logs, weights and a small capture) diagnosed it; the fixes
were then tested on the laptop. [measured] is from our runs, [reviewer] from the
review and not re-checked.

## Diagnosis
- **Not the training length, not the channel count.** Validation is flat across the
  whole learning-rate decay in all four runs; 16 and 24 channels tie. [measured]
- **The denoiser is limited by bias, not by noise.** On a static scene its error
  stops improving after 3 frames and then rises. [reviewer]
  - Kernels barely stop at edges (a neighbour across a depth or normal edge gets
    0.055 of the weight, one on the same surface 0.064).
  - No history-length input: the history weight is about 0.85 from the first frame.
  - The filtered output is the next frame's history and is filtered again.
- **Demodulation blew up next to emitters and on dark glossy surfaces.** Directly
  visible emission was inside the training input and divided by albedo + 0.01
  (values of several hundred); in the fireplace room those pixels carried 50-74% of
  the error. [reviewer] The renderer hands the denoiser its input without that
  emission, so training and runtime disagreed. [measured]
- **The loss was not exposure-normalized:** the validation number was mostly the
  bright scenes; night Bistro ended worse than its raw input. [measured]
- **The joint network's learned output pass learned three near-constants** (history
  clipping off, history 3-8 frames long), worth 0.24 dB. Its denoiser stage started
  from scratch and got 37k steps against the denoisers' 200k. [reviewer / measured]
- **Too little data:** 64 training sequences, each drawn about 2,300 times; training
  loss kept falling while validation stood still. [measured]
- **Capture bugs** [measured after the review pointed at them]:
  - the output-size tracer's shared jitter used every 33rd index of its sequence, so
    its vertical positions never reached the lower third of the pixel;
  - references shared one sub-pixel position per dispatch of 32 samples.
- **The missing baseline** [measured, small laptop capture]: the v4 denoiser + TAAU
  scores L1 0.0223 / 25.22 dB, the v4 joint network 0.0222 / 24.76 dB. The joint
  network did not beat the pipeline it was meant to replace.

## Changes (8e02502 and after)
Capture: `--capture-emission` (emission in its own channels, out of the input), the
output-size tracer's jitter follows its 1-spp frames, a random sub-pixel position per
reference sample, `--reference-frames N` (longer sequences with references only for
the last N frames).

Temporal denoiser, options of training/temporal/model.py (off by default, so today's
exported models are unchanged):
- `edge`: depth, normal and albedo edge stops in the kernel logits (learned scales);
- `length`: history length as an input and the running mean as the default blend;
- `feedback blend`: the unfiltered accumulation is the history;
- `hidden`: a ReLU layer in the head; `eps`: the demodulation floor;
- `--exposure`: exposure-normalized loss and features.
The fast network's trainer takes the same options and the emission channel
(training/fast/baseline.py measures it against a denoiser + TAAU).

## Ablation on the laptop [measured]
4 scenes x 8 sequences x 16 frames (640x360, 160-pixel windows, 256-spp finished
references, new capture format), 16 channels, 1200 steps each, exposure-normalized
loss everywhere. Held-out sequences of the three trained scenes, evaluate.py (log(1+x)):

| variant | L1 | PSNR | flicker | hold-out scene L1 / PSNR |
|---|---:|---:|---:|---:|
| base (the v4 architecture) | 0.0717 | 22.27 dB | 0.1337 | 0.0423 / 19.97 |
| demodulation floor 0.05 | 0.0664 | 22.76 | 0.1126 | 0.0424 / 19.61 |
| edge stops | 0.0660 | 23.34 | 0.1329 | 0.0395 / 20.20 |
| history length | 0.0718 | 22.24 | 0.1329 | 0.0421 / 19.88 |
| history length + unfiltered feedback | 0.0696 | 22.40 | 0.1345 | 0.0411 / 20.04 |
| all four | 0.0605 | 24.00 | 0.1091 | 0.0377 / 20.06 |
| all four + head layer of 16 | 0.0597 | 23.98 | 0.1094 | 0.0376 / 19.93 |
| all four, 24 channels | 0.0602 | 24.00 | 0.1089 | 0.0380 / 19.98 |

- All four together: +1.7 dB, 16% lower L1, 18% lower flicker than the v4
  architecture at equal training. The v4 retraining itself was worth +0.3 to +0.45 dB.
- The floor and the edge stops carry most of it; the history changes add little at
  16-frame sequences. A head layer and more channels add nothing.
- Small data, short CPU runs: the ranking is what counts, not the absolute numbers.
  The second half of the table trained with the exposure taken from a sequence's last
  frame instead of its first (a mid-run code change); the evaluation is the same for all.

## Not done
- The new options are not in export.py or the renderer's shaders. The edge stops read
  depth, normal and albedo of every kernel tap: a runtime cost to measure.
- Longer sequences and about ten times the data need the PC.
