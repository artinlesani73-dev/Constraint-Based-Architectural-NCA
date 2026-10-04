"""Isolated correction of the v31 vertical scan; all other target rules retained.

This operator deliberately retains legacy target-legality and routing defects.
Use the separately versioned legal router for that next comparison. Historical
code in deploy/model_utils.py stays untouched as the replay oracle.
"""
import torch
import torch.nn.functional as F
from deploy.model_utils import (_extract_access_centroids, _compute_mst_edges,
                                _compute_knn_edges, compute_distance_field_3d)

BOUNDED_VERSION = "corridor_bounded_v1"


def _radius(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Radius must be a nonnegative integer")


def bounded_vertical_envelope(field, radius):
    """Bounded depth-only maximum on (..., depth, height, width), without mutation.

    Leading axes are independent; accepts finite nonnegative floating fields.
    Radius zero returns a distinct tensor. The maximum reads only original rows.
    """
    _radius(radius)
    if field.ndim < 3 or any(n == 0 for n in field.shape):
        raise ValueError("Expected nonempty spatial and optional leading axes")
    if not field.is_floating_point() or not torch.isfinite(field).all() or (field < 0).any():
        raise ValueError("Expected finite nonnegative floating values")
    if radius == 0:
        return field.clone()
    # A radius larger than depth-1 already covers the whole depth dimension.
    radius = min(radius, field.shape[-3] - 1)
    pooled = F.max_pool3d(field.reshape(-1, 1, *field.shape[-3:]),
                         (2 * radius + 1, 1, 1), 1, (radius, 0, 0))
    return pooled.reshape(field.shape)


def compute_corridor_target_bounded_v1(seed_state: torch.Tensor, config: dict,
                                corridor_width: int = 1,
                                vertical_envelope: int = 1) -> torch.Tensor:
    _radius(corridor_width)
    _radius(vertical_envelope)
    cfg = config
    G = cfg['grid_size']
    device = seed_state.device
    B = seed_state.shape[0]

    corridors = torch.zeros(B, G, G, G, device=device)

    for b in range(B):
        access = seed_state[b, cfg['ch_access']]
        existing = seed_state[b, cfg['ch_existing']]
        legal_mask = 1.0 - existing

        centroids = _extract_access_centroids(access.detach().cpu())

        if len(centroids) < 2:
            dilated = F.max_pool3d(
                access.unsqueeze(0).unsqueeze(0),
                2 * corridor_width + 1,
                1,
                corridor_width,
            )
            corridors[b] = dilated.squeeze() * legal_mask
            continue

        corridor_mask = torch.zeros(G, G, G, device=device)
        edges = set(_compute_mst_edges(centroids))
        edges.update(_compute_knn_edges(centroids, k=1))

        for i_idx, j_idx in edges:
            start = centroids[i_idx]
            end = centroids[j_idx]

            dist_from_start = compute_distance_field_3d([start], legal_mask)
            dist_from_end = compute_distance_field_3d([end], legal_mask)
            total_dist = dist_from_start[end[0], end[1], end[2]]
            if total_dist == float('inf'):
                continue

            path_cost = dist_from_start + dist_from_end
            slack = corridor_width
            on_path = (path_cost <= total_dist + slack).float()
            corridor_mask = torch.max(corridor_mask, on_path)

        if corridor_mask.sum() > 0:
            corridor_4d = corridor_mask.unsqueeze(0).unsqueeze(0)
            dilated = F.max_pool3d(corridor_4d, 2 * corridor_width + 1, 1, corridor_width)
            corridor_dilated = dilated.squeeze()

            corridor_dilated = bounded_vertical_envelope(corridor_dilated, vertical_envelope)

            # Clamp corridor to a Z band around access points (MODEL C behavior)
            if centroids and config.get('corridor_z_margin', None) is not None:
                z_vals = [c[0] for c in centroids]
                z_min = max(0, min(z_vals) - config['corridor_z_margin'])
                z_max = min(G, max(z_vals) + config['corridor_z_margin'] + 1)
                z_mask = torch.zeros_like(corridor_dilated)
                z_mask[z_min:z_max] = 1.0
                corridor_dilated = corridor_dilated * z_mask

            corridors[b] = corridor_dilated * legal_mask

    return corridors
