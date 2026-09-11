from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    """Causal streaming transition decoder configuration.

    V5.2 keeps hysteresis/refractory suppression, disables any mandatory stage gate
    by default, and optionally uses the learned signed distance head to refine the
    timestamp of a *detected* event retrospectively.

    distance_correction_weight is validation-selected.  A correction is applied only
    when the distance head says the boundary is at or before the current trigger
    (predicted current-minus-boundary distance >= 0), so the detector never reports a
    future event timestamp as if it had already occurred.
    """
    enter_threshold: float = 0.50
    exit_threshold: float = 0.30
    refractory_sec: float = 1.0
    stage_gate_min: float = 0.0
    exit_consecutive: int = 2
    distance_correction_weight: float = 0.0
    max_distance_correction_sec: float = 120.0

    def to_dict(self):
        return asdict(self)


def causal_stage_dynamics(exp_ids, times, stage_prob, max_lag: int = 3):
    exp_ids = np.asarray(exp_ids, dtype=object)
    times = np.asarray(times, dtype=np.float64)
    p_all = np.asarray(stage_prob, dtype=np.float32)
    n = len(p_all)
    delta_out = np.zeros(n, dtype=np.float32)
    forward_out = np.zeros(n, dtype=np.float32)

    if p_all.ndim != 2 or p_all.shape[1] < 2:
        return delta_out, forward_out

    stage_axis = np.linspace(0.0, 1.0, p_all.shape[1], dtype=np.float32)
    for eid in np.unique(exp_ids):
        idx = np.where(exp_ids == eid)[0]
        order = idx[np.argsort(times[idx])]
        p = p_all[order]
        expected = p @ stage_axis
        delta = np.zeros(len(order), dtype=np.float32)
        forward = np.zeros(len(order), dtype=np.float32)
        for lag in range(1, max_lag + 1):
            if len(order) <= lag:
                continue
            tv = 0.5 * np.abs(p[lag:] - p[:-lag]).sum(axis=1)
            fw = expected[lag:] - expected[:-lag]
            delta[lag:] = np.maximum(delta[lag:], tv.astype(np.float32))
            forward[lag:] = np.maximum(forward[lag:], np.maximum(fw, 0.0).astype(np.float32))
        delta_out[order] = delta
        forward_out[order] = forward
    return delta_out, forward_out


def gated_transition_score(exp_ids, times, heat_prob, stage_prob, max_lag: int = 3):
    heat = np.asarray(heat_prob, dtype=np.float32)
    stage_delta, forward = causal_stage_dynamics(exp_ids, times, stage_prob, max_lag=max_lag)
    # Keep stage dynamics as evidence, but do not hard-gate the event head.
    base = np.sqrt(np.clip(heat * np.maximum(stage_delta, 1e-4), 0.0, 1.0)).astype(np.float32)
    return base, stage_delta, forward.astype(np.float32), base.copy()


class StreamingTransitionDecoder:
    def __init__(self, cfg: DecoderConfig):
        self.cfg = cfg
        self.active = False
        self.below_run = 0
        self.last_event_t = -np.inf

    def update(self, t: float, score: float, forward_shift: float):
        fired = False
        if not self.active:
            refractory_ok = (float(t) - float(self.last_event_t)) >= float(self.cfg.refractory_sec)
            gate_ok = float(forward_shift) >= float(self.cfg.stage_gate_min)
            if refractory_ok and gate_ok and float(score) >= float(self.cfg.enter_threshold):
                self.active = True
                self.below_run = 0
                self.last_event_t = float(t)
                fired = True
        else:
            if float(score) < float(self.cfg.exit_threshold):
                self.below_run += 1
                if self.below_run >= int(self.cfg.exit_consecutive):
                    self.active = False
                    self.below_run = 0
            else:
                self.below_run = 0
        return fired, self.active


def decode_streaming_events(times, scores, forward_shift, cfg: DecoderConfig, distance_sec=None):
    """Decode events and optionally refine their timestamps with signed distance.

    distance_sec is the model prediction of current_time - boundary_time.  Positive
    values therefore allow a causal retrospective correction t_hat = t - distance.
    Negative values are not used for timestamp correction because that would place the
    reported event in the future relative to the current trigger.
    """
    times = np.asarray(times, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    forward_shift = np.asarray(forward_shift, dtype=np.float64)
    distance_sec = None if distance_sec is None else np.asarray(distance_sec, dtype=np.float64)
    decoder = StreamingTransitionDecoder(cfg)
    events = []
    active_mask = np.zeros(len(times), dtype=bool)
    for i, (t, score, fw) in enumerate(zip(times, scores, forward_shift)):
        fired, active = decoder.update(float(t), float(score), float(fw))
        active_mask[i] = bool(active)
        if fired:
            event_t = float(t)
            if distance_sec is not None and i < len(distance_sec):
                d = float(distance_sec[i])
                if np.isfinite(d) and d >= 0.0 and cfg.distance_correction_weight > 0:
                    d = min(d, max(float(cfg.max_distance_correction_sec), 0.0))
                    event_t = float(t) - float(cfg.distance_correction_weight) * d
                    event_t = min(event_t, float(t))
            events.append(event_t)
    return np.asarray(events, dtype=np.float64), active_mask
