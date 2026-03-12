from . import BaseController
import numpy as np
import onnxruntime as ort
from pathlib import Path

CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
STEER_RANGE = [-2, 2]
LAT_ACCEL_COST_MULTIPLIER = 50.0

_session_cache = None

def _get_session():
    global _session_cache
    if _session_cache is None:
        model_path = str(Path(__file__).resolve().parent.parent / 'models' / 'tinyphysics.onnx')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        with open(model_path, 'rb') as f:
            _session_cache = ort.InferenceSession(f.read(), options, ['CPUExecutionProvider'])
    return _session_cache


class Controller(BaseController):
    """
    Enhanced PID+FF with CEM action-correction MPC.
    Improvements over mpc.py:
    - Discount factor (0.90) weights near-term predictions higher
    - CEM warm-starting from previous plan
    - More CEM iterations (7 vs 5)
    """

    def __init__(self):
        self.session = _get_session()
        self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)

        # PID+FF gains (same as mpc.py)
        self.kp = 0.10
        self.ki = 0.12
        self.kd = -0.10
        self.ff_gain = 0.30
        self.roll_comp = 0.60
        self.preview_blend = 0.30
        self.preview_steps = 5
        self.preview_decay = 0.85
        self.d_alpha = 0.25

        # CEM parameters
        self.N_KNOTS = 6
        self.N_POP = 48
        self.N_ELITE = 12
        self.N_ITER = 7        # More iterations for better convergence
        self.DISCOUNT = 0.90   # Weight near-term predictions higher
        self.REPLAN = 10        # More responsive to trajectory drift
        self.CEM_START = 21

        # State
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.d_filter = 0.0
        self.states_hist = []
        self.actions_hist = []
        self.lataccels_hist = []
        self.step_count = 0

        # MPC corrections
        self.planned_corrections = None
        self.plan_start_step = 0
        self.last_plan_step = 0
        self.prev_best_knots = None  # For warm-starting

    def _softmax_batch(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def _encode(self, value):
        return np.digitize(
            np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1]),
            self.bins, right=True)

    def _batched_simulate(self, all_corrections, target, current, state, future_plan, H, uniform_draws):
        N = all_corrections.shape[0]
        base_tokens = self._encode(np.array(self.lataccels_hist[-CONTEXT_LENGTH:]))
        base_states = np.array(self.states_hist[-CONTEXT_LENGTH:])
        base_actions = np.array(self.actions_hist[-(CONTEXT_LENGTH - 1):])

        tokens = np.tile(base_tokens, (N, 1)).copy()
        states_ctx = np.tile(base_states, (N, 1, 1))
        actions_ctx = np.tile(base_actions, (N, 1))

        cur = np.full(N, current, dtype=np.float64)
        sim_ei = np.full(N, self.error_integral, dtype=np.float64)
        sim_pe = np.full(N, self.prev_error, dtype=np.float64)
        sim_df = np.full(N, self.d_filter, dtype=np.float64)
        total_tracking = np.zeros(N, dtype=np.float64)
        total_jerk = np.zeros(N, dtype=np.float64)

        # Pre-compute targets and states from future_plan
        tgts = np.empty(H, dtype=np.float64)
        rolls = np.empty(H, dtype=np.float64)
        fp_v = np.empty(H, dtype=np.float64)
        fp_a = np.empty(H, dtype=np.float64)
        fp_lat_arr = np.array(future_plan.lataccel) if future_plan.lataccel else np.array([])
        fp_roll_arr = np.array(future_plan.roll_lataccel) if future_plan.roll_lataccel else np.array([])
        fp_v_arr = np.array(future_plan.v_ego) if future_plan.v_ego else np.array([])
        fp_a_arr = np.array(future_plan.a_ego) if future_plan.a_ego else np.array([])

        tgts[0] = target
        rolls[0] = state.roll_lataccel
        for step in range(1, H):
            idx = step - 1
            tgts[step] = fp_lat_arr[idx] if idx < len(fp_lat_arr) else (fp_lat_arr[-1] if len(fp_lat_arr) > 0 else target)
            rolls[step] = fp_roll_arr[idx] if idx < len(fp_roll_arr) else state.roll_lataccel
            fp_v[step] = fp_v_arr[idx] if idx < len(fp_v_arr) else state.v_ego
            fp_a[step] = fp_a_arr[idx] if idx < len(fp_a_arr) else state.a_ego

        # Pre-compute preview weights
        preview_weights = {}
        for step in range(H):
            fp_start = step if step > 0 else 0
            fp_end = len(fp_lat_arr)
            n_preview = min(self.preview_steps, fp_end - fp_start)
            if n_preview > 0:
                w = np.array([self.preview_decay ** j for j in range(n_preview)])
                w /= w.sum()
                preview_weights[step] = float(np.dot(w, fp_lat_arr[fp_start:fp_start + n_preview]))
            else:
                preview_weights[step] = None

        # Pre-allocate model input buffer
        states_in = np.zeros((N, CONTEXT_LENGTH, 4), dtype=np.float32)
        act_buf = np.zeros((N, CONTEXT_LENGTH), dtype=np.float64)
        act_buf[:, 1:] = actions_ctx
        st_buf = states_ctx.copy()
        tokens_i64 = tokens.astype(np.int64)

        # Pre-compute discount weights
        discounts = np.array([self.DISCOUNT ** step for step in range(H)])

        for step in range(H):
            tgt = tgts[step]
            roll = rolls[step]
            d = discounts[step]

            # PID+FF (vectorized)
            error = tgt - cur
            ed = error - sim_pe
            sim_pe = error.copy()
            sim_df = self.d_alpha * ed + (1 - self.d_alpha) * sim_df
            sim_ei += error

            pv = preview_weights[step]
            if pv is not None:
                blended = (1 - self.preview_blend) * tgt + self.preview_blend * pv
                compensated = blended - self.roll_comp * roll
                pid_action = compensated * self.ff_gain + self.kp * error + self.ki * sim_ei + self.kd * sim_df
            else:
                pid_action = self.kp * error + self.ki * sim_ei + self.kd * sim_df

            action = np.clip(pid_action + all_corrections[:, step], STEER_RANGE[0], STEER_RANGE[1])

            # Update rolling buffers
            act_buf[:, :-1] = act_buf[:, 1:]
            act_buf[:, -1] = action
            if step > 0:
                st_buf[:, :-1, :] = st_buf[:, 1:, :]
                st_buf[:, -1, 0] = roll
                st_buf[:, -1, 1] = fp_v[step]
                st_buf[:, -1, 2] = fp_a[step]

            states_in[:, :, 0] = act_buf
            states_in[:, :, 1:] = st_buf

            res = self.session.run(None, {
                'states': states_in,
                'tokens': tokens_i64
            })[0]

            logits = res[:, -1, :] / 0.8
            logits -= logits.max(axis=-1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

            cumprobs = np.cumsum(probs, axis=-1)
            token_samples = np.argmax(cumprobs >= uniform_draws[step], axis=1)

            pred = self.bins[token_samples].astype(np.float64)
            pred = np.clip(pred, cur - MAX_ACC_DELTA, cur + MAX_ACC_DELTA)

            tokens_i64[:, :-1] = tokens_i64[:, 1:]
            tokens_i64[:, -1] = self._encode(pred)

            # Discounted cost
            total_tracking += d * (tgt - pred) ** 2
            total_jerk += d * ((pred - cur) / DEL_T) ** 2
            cur = pred

        costs = total_tracking / H * 100 * LAT_ACCEL_COST_MULTIPLIER + total_jerk / H * 100
        return costs

    def _cem_plan(self, target, current, state, future_plan):
        rs = np.random.get_state()
        rng_state = (rs[0], rs[1].copy(), rs[2], rs[3], rs[4])

        n_avail = len(future_plan.lataccel) if future_plan.lataccel else 0
        H = min(40, n_avail)
        if H < 10:
            np.random.set_state(rng_state)
            return

        np.random.set_state(rng_state)
        uniform_draws = np.array([np.random.random() for _ in range(H)])

        knot_positions = np.linspace(0, H - 1, self.N_KNOTS)

        # Warm-start from previous plan if available
        if self.prev_best_knots is not None and self.planned_corrections is not None:
            shift = self.step_count - self.plan_start_step
            old_corr = self.planned_corrections
            if shift < len(old_corr):
                # Shift old corrections and sample at new knot positions
                remaining = old_corr[shift:]
                if len(remaining) >= 2:
                    n_old = min(len(remaining), H)
                    old_x = np.linspace(0, H - 1, n_old)
                    mean = np.interp(knot_positions, old_x, remaining[:n_old])
                else:
                    mean = np.zeros(self.N_KNOTS)
            else:
                mean = np.zeros(self.N_KNOTS)
            std = np.full(self.N_KNOTS, 0.15)  # Tighter std for warm-start
        else:
            mean = np.zeros(self.N_KNOTS)
            std = np.full(self.N_KNOTS, 0.20)

        best_cost = float('inf')
        best_knots = np.zeros(self.N_KNOTS)

        for iteration in range(self.N_ITER):
            pop_knots = []
            if iteration == 0:
                pop_knots.append(np.zeros(self.N_KNOTS))
                if self.prev_best_knots is not None:
                    pop_knots.append(mean.copy())  # Also try the warm-start
            while len(pop_knots) < self.N_POP:
                sample = mean + std * np.random.randn(self.N_KNOTS)
                pop_knots.append(np.clip(sample, -0.8, 0.8))

            all_corrections = np.array([
                np.interp(np.arange(H), knot_positions, knots)
                for knots in pop_knots
            ])

            costs = self._batched_simulate(
                all_corrections, target, current, state, future_plan, H, uniform_draws)

            for i in range(self.N_POP):
                if costs[i] < best_cost:
                    best_cost = costs[i]
                    best_knots = np.array(pop_knots[i]).copy()

            elite_idx = np.argsort(costs)[:self.N_ELITE]
            elites = np.array([pop_knots[i] for i in elite_idx])
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), 0.002)

        self.planned_corrections = np.interp(np.arange(H), knot_positions, best_knots)
        self.plan_start_step = self.step_count
        self.prev_best_knots = best_knots.copy()

        np.random.set_state(rng_state)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.step_count += 1
        self.states_hist.append([state.roll_lataccel, state.v_ego, state.a_ego])
        self.lataccels_hist.append(current_lataccel)

        if (self.step_count == self.CEM_START or
            (self.step_count > self.CEM_START and
             self.step_count - self.last_plan_step >= self.REPLAN)):
            self._cem_plan(target_lataccel, current_lataccel, state, future_plan)
            self.last_plan_step = self.step_count

        # Compute PID+FF action
        error = target_lataccel - current_lataccel
        error_diff = error - self.prev_error
        self.d_filter = self.d_alpha * error_diff + (1 - self.d_alpha) * self.d_filter

        ff = 0.0
        if future_plan.lataccel and len(future_plan.lataccel) > 0:
            n = min(self.preview_steps, len(future_plan.lataccel))
            w = np.array([self.preview_decay ** j for j in range(n)])
            w /= w.sum()
            preview = float(np.dot(w, np.array(future_plan.lataccel[:n])))
            blended = (1 - self.preview_blend) * target_lataccel + self.preview_blend * preview
            compensated = blended - self.roll_comp * state.roll_lataccel
            ff = compensated * self.ff_gain

        self.error_integral += error
        fb = self.kp * error + self.ki * self.error_integral + self.kd * self.d_filter
        action = ff + fb

        # Apply planned correction
        if self.planned_corrections is not None:
            corr_idx = self.step_count - self.plan_start_step
            if 0 <= corr_idx < len(self.planned_corrections):
                action += self.planned_corrections[corr_idx]

        self.prev_error = error
        clipped = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
        self.actions_hist.append(clipped)
        return clipped
