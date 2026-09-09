from __future__ import annotations

from typing import Any


def aggregate_undirected_axes(
    candidates: Any, weights: Any, valid: Any, torch: Any, *,
    return_moment: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    """Aggregate unoriented candidate axes through a weighted second moment."""
    safe = torch.nn.functional.normalize(
        candidates.nan_to_num(posinf=0.0, neginf=0.0), dim=-1, eps=1e-6
    )
    normalized = torch.softmax(weights.masked_fill(~valid, -1e4), dim=-1)
    normalized = normalized * valid.to(normalized.dtype)
    normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    moment = torch.einsum("...k,...ki,...kj->...ij", normalized, safe, safe)
    eigenvalues, eigenvectors = torch.linalg.eigh(moment)
    axis = eigenvectors[..., -1]
    coverage = valid.any(dim=-1)
    # A zero vector explicitly represents low observability. It cannot inject a
    # preferred world direction into downstream losses or visualization.
    axis = torch.where(coverage[..., None], axis, torch.zeros_like(axis))
    confidence = torch.where(
        coverage,
        (eigenvalues[..., -1] - eigenvalues[..., -2]).clamp_min(0.0),
        torch.zeros_like(eigenvalues[..., -1]),
    )
    return (axis, confidence, moment) if return_moment else (axis, confidence)


def build_equivariant_pair_geometry(
    trajectory_tokens: Any,
    trajectory_visibility: Any,
    slot_probabilities: Any,
    torch: Any,
    *,
    minimum_effective_tracks: float = 2.5,
) -> dict[str, Any]:
    """Build differentiable scalar/vector pair evidence from lifted tracks.

    Every vector is formed only from positions, translations, cross products,
    and Kabsch rotation logs. Therefore a shared proper rotation Q maps every
    returned vector v to Qv while all scalar diagnostics remain invariant.
    """
    reference = trajectory_tokens[..., 0:3]
    positions = reference + trajectory_tokens[..., 3:6]
    visible = trajectory_visibility.to(positions.dtype)
    slot_weights = slot_probabilities.transpose(1, 2)
    batch, slots, tracks = slot_weights.shape
    frames = positions.shape[2]

    frame_weights = slot_weights[..., None] * visible[:, None, :, :]
    weight_sum = frame_weights.sum(dim=2).clamp_min(1e-6)
    centroids = torch.einsum("bktf,btfd->bkfd", frame_weights, positions)
    centroids = centroids / weight_sum[..., None]
    reference_centers = torch.einsum(
        "bkt,btd->bkd", slot_weights, reference[:, :, 0]
    ) / slot_weights.sum(dim=2, keepdim=True).clamp_min(1e-6)

    # Batched weighted Kabsch. The previous implementation executed one Python
    # SVD for every batch x slot x frame tuple, leaving the GPU mostly idle.
    total_weight = frame_weights.sum(dim=2)
    squared_weight = frame_weights.square().sum(dim=2)
    effective_count = total_weight.square() / squared_weight.clamp_min(1e-6)
    safe_total = total_weight.clamp_min(1e-6)
    source_centers = torch.einsum(
        "bktf,btd->bkfd", frame_weights, reference[:, :, 0]
    ) / safe_total[..., None]
    target_centers = centroids
    cross_moment = torch.einsum(
        "bktf,bti,btfj->bkfij", frame_weights, reference[:, :, 0], positions
    )
    covariance = cross_moment - safe_total[..., None, None] * (
        source_centers[..., :, None] * target_centers[..., None, :]
    )
    eye = torch.eye(3, dtype=positions.dtype, device=positions.device)
    covariance = covariance + eye * 1e-8
    u, _, vh = torch.linalg.svd(covariance)
    correction = eye.expand(batch, slots, frames, 3, 3).clone()
    correction[..., -1, -1] = torch.det(vh.transpose(-1, -2) @ u.transpose(-1, -2))
    rotation = vh.transpose(-1, -2) @ correction @ u.transpose(-1, -2)
    translation = target_centers - torch.einsum(
        "bkfij,bkfj->bkfi", rotation, source_centers
    )
    replay = torch.einsum(
        "bkfij,btj->bktfi", rotation, reference[:, :, 0]
    ) + translation[:, :, None, :, :]
    squared_error = (replay - positions[:, None]).square().sum(dim=-1)
    residual = torch.sqrt(
        (frame_weights * squared_error).sum(dim=2) / safe_total
    )
    sufficiently_observed = effective_count >= minimum_effective_tracks
    transforms_r = torch.where(
        sufficiently_observed[..., None, None], rotation, eye
    )
    transforms_t = torch.where(
        sufficiently_observed[..., None], translation, torch.zeros_like(translation)
    )
    rigid_residual = torch.where(
        sufficiently_observed, residual, torch.full_like(residual, float("inf"))
    )

    candidate_axes = positions.new_zeros((batch, slots, slots, 4, 3))
    candidate_valid = torch.zeros(
        (batch, slots, slots, 4), dtype=torch.bool, device=positions.device
    )
    candidate_pivots = positions.new_zeros((batch, slots, slots, 4, 3))
    candidate_residual = positions.new_full(
        (batch, slots, slots, 4), float("inf")
    )
    observability = positions.new_zeros((batch, slots, slots))
    pair_centers = 0.5 * (
        reference_centers[:, :, None, :] + reference_centers[:, None, :, :]
    )
    pivot_basis_vectors = positions.new_zeros((batch, slots, slots, 2, 3))

    for parent in range(slots):
        for child in range(slots):
            if parent == child:
                continue
            rp = transforms_r[:, parent]
            rc = transforms_r[:, child]
            tp = transforms_t[:, parent]
            tc = transforms_t[:, child]
            # The transform mapping the fitted parent pose to the fitted child
            # pose is expressed in the shared world/source frame. Under a
            # global basis rotation Q it becomes (Q R Q^T, Q t), so both its
            # rotation-log axis and hinge line are strictly equivariant.
            relative_r = rc @ rp.transpose(-1, -2)
            relative_t = tc - torch.einsum("bfij,bfj->bfi", relative_r, tp)
            valid_frames = (
                (effective_count[:, parent] >= minimum_effective_tracks)
                & (effective_count[:, child] >= minimum_effective_tracks)
            )
            rotation_vectors, rotation_angle = _rotation_log(relative_r, torch)
            rotation_valid = valid_frames & (rotation_angle > 1e-3)

            # Candidate 0: full-path relative translation PCA (pull-return safe).
            relative_centroid_path = centroids[:, child] - centroids[:, parent]
            centered_t = relative_centroid_path - _masked_mean(
                relative_centroid_path, valid_frames, torch
            )
            covariance = torch.einsum(
                "bf,bfi,bfj->bij", valid_frames.to(centered_t.dtype), centered_t, centered_t
            )
            _, translation_vectors = torch.linalg.eigh(covariance)
            translation_axis = translation_vectors[..., -1]
            translation_extent = torch.linalg.vector_norm(centered_t, dim=-1).amax(dim=-1)
            candidate_axes[:, parent, child, 0] = translation_axis
            candidate_valid[:, parent, child, 0] = (
                (valid_frames.sum(dim=-1) >= 2) & (translation_extent > 1e-4)
            )
            candidate_residual[:, parent, child, 0] = _axis_projection_residual(
                centered_t, translation_axis, valid_frames, torch
            )

            # Candidate 1: sign-invariant aggregation of relative-SE(3) rotation logs.
            rotation_axes = torch.nn.functional.normalize(rotation_vectors, dim=-1, eps=1e-6)
            rotation_axis, rotation_confidence = aggregate_undirected_axes(
                rotation_axes,
                rotation_angle,
                rotation_valid,
                torch,
            )
            candidate_axes[:, parent, child, 1] = rotation_axis
            candidate_valid[:, parent, child, 1] = rotation_valid.any(dim=-1)
            rotation_dot = torch.sum(rotation_axes * rotation_axis[:, None, :], dim=-1).abs()
            candidate_residual[:, parent, child, 1] = _masked_mean_scalar(
                1.0 - rotation_dot, rotation_valid, torch
            )

            # Candidate 2/3 supply equivariant geometric alternatives to the VN head.
            center_delta = reference_centers[:, child] - reference_centers[:, parent]
            peak_index = torch.linalg.vector_norm(centered_t, dim=-1).argmax(dim=-1)
            peak_motion = torch.gather(
                centered_t, 1, peak_index[:, None, None].expand(-1, 1, 3)
            )[:, 0]
            # Pure VN pivot primitives: both are oriented vectors constructed
            # directly from points and full trajectories, without hinge fitting.
            pivot_basis_vectors[:, parent, child, 0] = center_delta
            pivot_basis_vectors[:, parent, child, 1] = peak_motion
            tangent_normal = torch.linalg.cross(center_delta, translation_axis, dim=-1)
            candidate_axes[:, parent, child, 2] = center_delta
            candidate_axes[:, parent, child, 3] = tangent_normal
            candidate_valid[:, parent, child, 2] = (
                torch.linalg.vector_norm(center_delta, dim=-1) > 1e-5
            )
            candidate_valid[:, parent, child, 3] = (
                (torch.linalg.vector_norm(tangent_normal, dim=-1) > 1e-5)
                & candidate_valid[:, parent, child, 0]
            )
            candidate_residual[:, parent, child, 2:] = candidate_residual[
                :, parent, child, :2
            ]

            pivot = _hinge_line_point(
                relative_r, relative_t, rotation_valid, rotation_axis, torch
            )
            candidate_pivots[:, parent, child] = pivot[:, None, :]
            observability[:, parent, child] = torch.maximum(
                translation_extent,
                rotation_angle.masked_fill(~rotation_valid, 0.0).amax(dim=-1),
            ) * torch.maximum(rotation_confidence, positions.new_tensor(0.1))

    scalar_features = torch.stack(
        [
            candidate_residual[..., 0].nan_to_num(posinf=1e3),
            candidate_residual[..., 1].nan_to_num(posinf=1e3),
            observability,
            effective_count.mean(dim=-1)[:, :, None].expand(-1, -1, slots),
            effective_count.mean(dim=-1)[:, None, :].expand(-1, slots, -1),
        ],
        dim=-1,
    )
    # Kabsch/SVD/lstsq are analytic proposal generators rather than learned
    # layers. Their derivatives are undefined at common rigid-motion
    # degeneracies (repeated singular/eigen values), which can inject NaNs into
    # otherwise unrelated edge/type branches. Learn only the scalar proposal
    # weights; slot fine-tuning still receives gradients through slot_pair and
    # the ordinary assignment losses.
    return {
        "candidate_axes": candidate_axes.detach(),
        "candidate_valid": candidate_valid,
        "candidate_pivots": candidate_pivots.nan_to_num().detach(),
        "candidate_residual": candidate_residual.detach(),
        "scalar_features": scalar_features.detach(),
        "pair_centers": pair_centers.detach(),
        "pivot_basis_vectors": pivot_basis_vectors.detach(),
        "observability": observability.detach(),
    }


def _rotation_log(rotation: Any, torch: Any) -> tuple[Any, Any]:
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    skew = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    sine = torch.sin(angle)
    scale = torch.where(
        angle > 1e-5,
        angle / (2.0 * sine.clamp_min(1e-6)),
        torch.full_like(angle, 0.5),
    )
    return skew * scale[..., None], angle


def _hinge_line_point(
    relative_r: Any, relative_t: Any, valid: Any, axis: Any, torch: Any
) -> Any:
    batch = relative_r.shape[0]
    result = relative_t.new_zeros((batch, 3))
    eye = torch.eye(3, dtype=relative_r.dtype, device=relative_r.device)
    for index in range(batch):
        if int(valid[index].sum()) < 1:
            continue
        matrix = (eye - relative_r[index])[valid[index]]
        rhs = relative_t[index][valid[index]]
        # (I-R)p=t is rank deficient along the hinge axis. Explicitly choose
        # the closest point on the line to the origin instead of relying on a
        # coordinate-sensitive numerical rank decision inside lstsq.
        matrix = torch.cat([matrix.reshape(-1, 3), axis[index][None]], dim=0)
        rhs = torch.cat([rhs.reshape(-1, 1), rhs.new_zeros((1, 1))], dim=0)
        result[index] = torch.linalg.lstsq(
            matrix, rhs
        ).solution[:, 0]
    return result


def _masked_mean(values: Any, valid: Any, torch: Any) -> Any:
    weights = valid.to(values.dtype)
    return (
        (values * weights[..., None]).sum(dim=-2, keepdim=True)
        / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)[..., None]
    )


def _masked_mean_scalar(values: Any, valid: Any, torch: Any) -> Any:
    weights = valid.to(values.dtype)
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def _axis_projection_residual(values: Any, axis: Any, valid: Any, torch: Any) -> Any:
    projection = torch.sum(values * axis[:, None, :], dim=-1, keepdim=True) * axis[:, None, :]
    residual = torch.linalg.vector_norm(values - projection, dim=-1)
    return _masked_mean_scalar(residual, valid, torch)
