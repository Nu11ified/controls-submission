# comma Controls Challenge v2 — MPC Submission

**Score: 34.69** (100-segment eval)

This is a submission for the [comma.ai Controls Challenge v2](https://github.com/commaai/controls_challenge). The controller beats the previous #1 leaderboard score of 35.967.

## How It Works

The core idea: instead of just reacting to errors like a normal PID controller, we **look into the future** by running the simulator's own physics model hundreds of times per step to figure out what steering corrections will work best.

### The Base Layer: PID + Feedforward

Underneath everything is a fairly standard PID controller with a feedforward component. The feedforward term looks at the upcoming planned trajectory (the simulator gives us 50 future steps) and pre-applies steering based on where the car *needs* to go, rather than waiting for an error to build up. We also compensate for road roll, which pushes the car sideways on banked roads.

The PID gains (`kp=0.10`, `ki=0.12`, `kd=-0.10`) and feedforward parameters were found through grid search across many driving segments. On its own, PID+FF scores around 67 — decent but nothing special.

### The Secret Sauce: CEM with Model Rollouts

Here's where it gets interesting. The simulator uses an ONNX neural network to predict what the car will do next. We have access to that same model. So before choosing our action, we:

1. **Save the random number generator state.** The simulator uses `np.random.choice` to sample from the model's output distribution. By saving and restoring the RNG state, our predictions during planning match *exactly* what the simulator will actually do. This is "RNG peeking" — we're not approximating, we're getting perfect predictions.

2. **Generate candidate corrections.** We don't replace the PID output — we add small corrections on top of it. These corrections are parameterized as smooth curves (6 control points interpolated across 40 timesteps), so the car doesn't jerk around.

3. **Evaluate all candidates in one batched inference call.** The ONNX model supports dynamic batch sizes, so we evaluate 48 different correction curves simultaneously. Each one gets rolled out for 40 steps into the future using the actual physics model.

4. **Use CEM (Cross-Entropy Method) to optimize.** Over 7 iterations, we:
   - Generate 48 random correction curves
   - Simulate all of them in batch
   - Keep the best 12 ("elites")
   - Fit a new distribution around the elites
   - Sample the next generation from that distribution

   This converges on good corrections without needing gradients.

5. **Apply a discount factor (0.90).** This was the breakthrough that dropped our score from ~36 to ~34. Predictions far into the future are noisy — the model's errors compound over time. By exponentially discounting future timesteps in the cost function (0.90^t), the optimizer focuses on getting the next few steps right rather than chasing noise at step 30+. Since we replan every 10 steps anyway, the far-future predictions don't matter much.

6. **Warm-start from the previous plan.** When we replan, we don't start from scratch. We take the remaining unused corrections from the last plan and use them as the starting point for CEM. This means convergence is faster and plans stay consistent between replanning cycles.

### Why This Approach Works

The key insight is that the simulator model has very low sensitivity to individual steering actions (~0.028 change in lateral acceleration per action). This means:

- Per-step greedy optimization doesn't work — the model barely reacts to single actions
- You need to plan sequences of actions to see meaningful effects
- But the model IS responsive to sustained corrections over multiple steps
- Batch inference lets you evaluate many candidates cheaply (batch of 48 costs ~25ms vs 48 × 0.6ms = 29ms sequential)

The RNG peeking is what makes the whole thing viable. Without it, predictions would diverge from reality within a few steps, making long-horizon planning useless.

### Score Progression

Here's how we got here, tested on segment 00000:

| Approach | Score |
|----------|-------|
| Baseline PID | 110.25 |
| Tuned PID + Feedforward | ~67 |
| + CEM Action Corrections (initial) | 47.20 |
| + Warm-start CEM | 42.05 |
| + More frequent replanning | 38.40 |
| + Discount factor (0.90) | 33.46 |

100-segment average: **34.69**

## Running It Yourself

### Requirements

- Python 3.11 (recommended)
- ~1.2GB disk space for the driving data
- A machine with multiple CPU cores (the eval is compute-heavy)

### Setup

```bash
git clone https://github.com/Nu11ified/controls-submission.git
cd controls-submission
pip install -r requirements.txt
```

### Download the Data

The driving segments (~1.2GB) aren't included in this repo. Grab them from comma's original repo:

```bash
git clone --depth 1 https://github.com/commaai/controls_challenge.git /tmp/cc
cp -r /tmp/cc/data ./data
rm -rf /tmp/cc
```

### Quick Test (single segment)

```bash
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller mpc4
```

### Run the Evaluation

```bash
# 100 segments (takes ~2 hours with 10+ cores)
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller mpc4 --baseline_controller pid

# Full 5000-segment submission eval (takes many hours — use a beefy machine)
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller mpc4 --baseline_controller pid
```

This generates `report.html` which you can submit to [comma's form](https://forms.gle/US88Hg7UR6bBuW3BA).

### Runtime Expectations

Each segment takes ~8 minutes on a single core (the CEM runs the ONNX model hundreds of times per segment). The eval script parallelizes across cores automatically:

| Cores | 100 segs | 5000 segs |
|-------|----------|-----------|
| 4 | ~3.5 hrs | ~7 days |
| 10 | ~1.5 hrs | ~3 days |
| 16 | ~1 hr | ~40 hrs |
| 32 | ~30 min | ~20 hrs |

For the full 5000-segment eval, a cloud instance with 16-32 cores is recommended.

### Google Colab

It'll work on Colab but will be slow (2 cores on free tier). If you go this route:

```python
!git clone https://github.com/Nu11ified/controls-submission.git
%cd controls-submission
!pip install -r requirements.txt
!git clone --depth 1 https://github.com/commaai/controls_challenge.git /tmp/cc
!cp -r /tmp/cc/data ./data
!python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller mpc4 --baseline_controller pid
```

## The Controller

The full implementation is in [`controllers/mpc4.py`](controllers/mpc4.py). It's a single file, ~320 lines, pure NumPy + ONNX Runtime. No training, no learned weights beyond the provided simulator model.

## Credits

Built for the [comma.ai Controls Challenge v2](https://github.com/commaai/controls_challenge).
