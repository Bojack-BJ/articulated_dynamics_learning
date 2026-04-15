from __future__ import annotations

from statistics import fmean

from .models import ArticulationArtifact, EpisodeInput, StateSample


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def smooth_sequence(values: list[float], window_size: int = 3) -> list[float]:
    if window_size <= 1 or len(values) < 2:
        return list(values)
    radius = window_size // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        smoothed.append(fmean(values[start:stop]))
    return smoothed


def differentiate_joint_signal(times: list[float], values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    qdots: list[float] = []
    for index in range(len(values)):
        if index == 0:
            dt = max(times[1] - times[0], 1e-6)
            qdot = (values[1] - values[0]) / dt
        elif index == len(values) - 1:
            dt = max(times[-1] - times[-2], 1e-6)
            qdot = (values[-1] - values[-2]) / dt
        else:
            dt = max(times[index + 1] - times[index - 1], 1e-6)
            qdot = (values[index + 1] - values[index - 1]) / dt
        qdots.append(qdot)
    return qdots


def extract_joint_signal(episode: EpisodeInput, limits: list[float]) -> list[float]:
    lower, upper = limits
    current = lower
    signal: list[float] = []
    for frame in episode.frames:
        if frame.joint_position_hint is not None:
            current = frame.joint_position_hint
        elif "joint_position" in frame.action_log:
            current = float(frame.action_log["joint_position"])
        elif "target_open_fraction" in frame.action_log:
            fraction = clamp(float(frame.action_log["target_open_fraction"]), 0.0, 1.0)
            current = lower + fraction * (upper - lower)
        elif "command_delta" in frame.action_log:
            current = current + float(frame.action_log["command_delta"])
        signal.append(clamp(current, lower, upper))
    if not signal:
        return []
    if all(signal[0] == item for item in signal):
        span = upper - lower
        if span > 0.0 and len(signal) > 1:
            signal = [lower + span * index / (len(signal) - 1) for index in range(len(signal))]
    return signal


class SlidingWindowRefitter:
    def __init__(self, window_size: int = 3, confidence_floor: float = 0.35) -> None:
        self.window_size = window_size
        self.confidence_floor = confidence_floor

    def refit(self, episode: EpisodeInput, articulation: ArticulationArtifact) -> ArticulationArtifact:
        if not articulation.joints:
            raise ValueError("Articulation must contain at least one joint for temporal refit.")
        joint = articulation.joints[0]
        raw_q = extract_joint_signal(episode, joint.limits)
        smoothed_q = [clamp(value, joint.limits[0], joint.limits[1]) for value in smooth_sequence(raw_q, self.window_size)]
        times = [frame.timestamp_s for frame in episode.frames]
        qdots = differentiate_joint_signal(times, smoothed_q)

        state: list[StateSample] = []
        for frame, q_value, qdot_value in zip(episode.frames, smoothed_q, qdots):
            state.append(
                StateSample(
                    timestamp_s=frame.timestamp_s,
                    q=q_value,
                    qdot=qdot_value,
                    confidence=min(frame.observation_confidence, joint.confidence),
                )
            )
        articulation.state = state
        articulation.fit_metrics.update(
            {
                "temporal_optimizer": "sliding-window-smoother",
                "raw_signal_min": min(raw_q) if raw_q else 0.0,
                "raw_signal_max": max(raw_q) if raw_q else 0.0,
                "smoothed_signal_min": min(smoothed_q) if smoothed_q else 0.0,
                "smoothed_signal_max": max(smoothed_q) if smoothed_q else 0.0,
                "temporal_smoothness_residual": self._smoothness_residual(smoothed_q),
            }
        )
        if articulation.joints[0].confidence < self.confidence_floor:
            articulation.low_confidence_components.append(articulation.joints[0].name)
        return articulation

    def _smoothness_residual(self, values: list[float]) -> float:
        if len(values) < 3:
            return 0.0
        residuals = []
        for index in range(1, len(values) - 1):
            residuals.append(abs(values[index + 1] - 2 * values[index] + values[index - 1]))
        return fmean(residuals) if residuals else 0.0
