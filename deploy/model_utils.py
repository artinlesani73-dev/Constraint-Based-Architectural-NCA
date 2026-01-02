import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Tuple, List
from itertools import combinations

def _pairwise_dist(points: List[Tuple[int, int, int]]) -> List[List[float]]:
    n = len(points)
    if n == 0:
        return []
    dists = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        zi, yi, xi = points[i]
        for j in range(i + 1, n):
            zj, yj, xj = points[j]
            d = abs(zi - zj) + abs(yi - yj) + abs(xi - xj)
            dists[i][j] = d
            dists[j][i] = d
    return dists


def _compute_mst_edges(points: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    n = len(points)
    if n < 2:
        return []
    dists = _pairwise_dist(points)
    in_tree = [False] * n
    in_tree[0] = True
    edges = []
    for _ in range(n - 1):
        best = None
        best_i = -1
        best_j = -1
        for i in range(n):
            if not in_tree[i]:
                continue
            for j in range(n):
                if in_tree[j]:
                    continue
                d = dists[i][j]
                if best is None or d < best:
                    best = d
                    best_i = i
                    best_j = j
        if best_j == -1:
            break
        in_tree[best_j] = True
        edges.append((best_i, best_j))
    return edges


def _compute_knn_edges(points: List[Tuple[int, int, int]], k: int = 1) -> List[Tuple[int, int]]:
    n = len(points)
    if n < 2:
        return []
    dists = _pairwise_dist(points)
    edges = set()
    for i in range(n):
        neighbors = sorted(
            [(dists[i][j], j) for j in range(n) if j != i],
            key=lambda x: x[0],
        )
        for _, j in neighbors[:k]:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return list(edges)


def _extract_access_centroids(access_mask: torch.Tensor) -> List[Tuple[int, int, int]]:
    D, H, W = access_mask.shape
    visited = set()
    centroids = []

    def neighbors(z: int, y: int, x: int):
        for dz, dy, dx in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                yield nz, ny, nx

    for z in range(D):
        for y in range(H):
            for x in range(W):
                if access_mask[z, y, x] < 0.5:
                    continue
                if (z, y, x) in visited:
                    continue
                stack = [(z, y, x)]
                vox = []
                visited.add((z, y, x))
                while stack:
                    cz, cy, cx = stack.pop()
                    vox.append((cz, cy, cx))
                    for nz, ny, nx in neighbors(cz, cy, cx):
                        if access_mask[nz, ny, nx] < 0.5:
                            continue
                        if (nz, ny, nx) in visited:
                            continue
                        visited.add((nz, ny, nx))
                        stack.append((nz, ny, nx))
                if vox:
                    zz = sum(v[0] for v in vox) / len(vox)
                    yy = sum(v[1] for v in vox) / len(vox)
                    xx = sum(v[2] for v in vox) / len(vox)
                    centroids.append((int(round(zz)), int(round(yy)), int(round(xx))))
    return centroids


def compute_distance_field_3d(start_points: List[Tuple[int, int, int]],
                              legal_mask: torch.Tensor,
                              max_iters: int = 64) -> torch.Tensor:
    D, H, W = legal_mask.shape
    device = legal_mask.device
    distance = torch.full((D, H, W), float('inf'), device=device)

    for z, y, x in start_points:
        if 0 <= z < D and 0 <= y < H and 0 <= x < W:
            distance[z, y, x] = 0

    for _ in range(max_iters):
        dist_4d = distance.unsqueeze(0).unsqueeze(0)
        expanded = -F.max_pool3d(-dist_4d, 3, 1, 1).squeeze(0).squeeze(0)
        expanded = expanded + 1

        new_distance = torch.where(
            legal_mask > 0.5,
            torch.min(distance, expanded),
            distance,
        )

        if torch.allclose(distance, new_distance, atol=1e-5):
            break
        distance = new_distance

    return distance


def compute_corridor_target_v31(seed_state: torch.Tensor, config: dict,
                                corridor_width: int = 1,
                                vertical_envelope: int = 1) -> torch.Tensor:
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

            if vertical_envelope > 0:
                for z in range(G):
                    z_min = max(0, z - vertical_envelope)
                    z_max = min(G, z + vertical_envelope + 1)
                    if corridor_dilated[z_min:z_max].max(dim=0)[0].any():
                        corridor_dilated[z] = torch.max(
                            corridor_dilated[z],
                            corridor_dilated[z_min:z_max].max(dim=0)[0],
                        )

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

class LocalLegalityLoss(nn.Module):
    """Per-voxel legality enforcement."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.street_levels = config['street_levels']

    def compute_legality_field(self, state: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        G = cfg['grid_size']
        street_levels = self.street_levels
        existing = state[:, cfg['ch_existing']]
        anchors = state[:, cfg['ch_anchors']]
        B = state.shape[0]
        device = state.device
        z_indices = torch.arange(G, device=device).view(1, G, 1, 1).expand(B, G, G, G)
        above_street = (z_indices >= street_levels).float()
        at_street = (z_indices < street_levels).float()
        position_legality = above_street + at_street * anchors
        legality = (1 - existing) * position_legality
        return torch.clamp(legality, 0, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        structure = state[:, cfg['ch_structure']]
        legality = self.compute_legality_field(state)
        illegal_structure = structure * (1 - legality)
        return illegal_structure.sum() / (structure.sum() + 1e-8)

class Perceive3D(nn.Module):
    def __init__(self, n_channels: int = 8):
        super().__init__()
        self.n_channels = n_channels
        sobel_x = self._create_sobel_kernel('x')
        sobel_y = self._create_sobel_kernel('y')
        sobel_z = self._create_sobel_kernel('z')
        identity = self._create_identity_kernel()
        kernels = torch.stack([identity, sobel_x, sobel_y, sobel_z], dim=0)
        self.register_buffer('kernels', kernels)

    def _create_sobel_kernel(self, direction: str) -> torch.Tensor:
        derivative = torch.tensor([-1., 0., 1.])
        smoothing = torch.tensor([1., 2., 1.])
        if direction == 'x':
            kernel = torch.einsum('i,j,k->ijk', smoothing, smoothing, derivative)
        elif direction == 'y':
            kernel = torch.einsum('i,j,k->ijk', smoothing, derivative, smoothing)
        elif direction == 'z':
            kernel = torch.einsum('i,j,k->ijk', derivative, smoothing, smoothing)
        return kernel / 16.0

    def _create_identity_kernel(self) -> torch.Tensor:
        kernel = torch.zeros(3, 3, 3)
        kernel[1, 1, 1] = 1.0
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x_padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
        outputs = []
        for k in range(4):
            kernel = self.kernels[k:k+1].unsqueeze(0).expand(C, 1, 3, 3, 3)
            out = F.conv3d(x_padded, kernel, padding=0, groups=C)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

class UrbanPavilionNCA(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        n_channels = config['n_channels']
        hidden_dim = config['hidden_dim']
        perception_dim = n_channels * 4
        n_grown = config['n_grown']

        self.perceive = Perceive3D(n_channels)
        self.update_net = nn.Sequential(
            nn.Conv3d(perception_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, n_grown, 1),
        )

    def forward(self, state: torch.Tensor, steps: int = 1) -> torch.Tensor:
        for _ in range(steps):
            state = self._step(state)
        return state

    def _step(self, state: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = state.shape
        cfg = self.config
        perception = self.perceive(state)
        delta = self.update_net(perception)

        # Only apply fire rate during training (original behavior)
        if self.training:
            fire_mask = (torch.rand(B, 1, D, H, W, device=state.device) < cfg['fire_rate']).float()
            delta = delta * fire_mask

        grown_start = cfg['n_frozen']
        grown_new = state[:, grown_start:] + cfg['update_scale'] * delta
        grown_new = torch.clamp(grown_new, 0.0, 1.0)

        existing = state[:, cfg['ch_existing']:cfg['ch_existing']+1]
        available_mask = 1.0 - existing
        struct_new = grown_new[:, 0:1] * available_mask

        # Online legality masking (v3.1 feature)
        if 'ch_anchors' in cfg:
            legality_field = LocalLegalityLoss(cfg).compute_legality_field(state)
            struct_new = struct_new * legality_field.unsqueeze(1)

        grown_masked = torch.cat([struct_new, grown_new[:, 1:]], dim=1)
        return torch.cat([state[:, :grown_start], grown_masked], dim=1)

    def grow(self, seed: torch.Tensor, steps: int = 50) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(seed, steps)

class UrbanSceneGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.G = config['grid_size']
        self.C = config['n_channels']

    def generate(self, params: dict, device: str = 'cpu') -> Tuple[torch.Tensor, dict]:
        G = self.G
        cfg = self.config
        state = torch.zeros(1, self.C, G, G, G, device=device)
        state[:, cfg['ch_ground'], 0, :, :] = 1.0

        building_info = self._place_buildings(state, params)
        access_info = self._place_access_points(state, params, building_info)
        anchor_info = self._generate_anchor_zones(state, params, building_info, access_info)
        return state, {
            'buildings': building_info, 
            'access_points': access_info,
            'anchor_zones': anchor_info
        }

    def _place_buildings(self, state: torch.Tensor, params: dict) -> list:
        G = self.G
        ch = self.config['ch_existing']
        buildings = []
        
        # Simplified/adapted placement based on parameters
        # In a real tool, the user would likely provide coordinates, 
        # but here we'll use the logic from the notebook to place them.
        
        buildings_data = params.get('buildings', [])
        for b in buildings_data:
            x_start, x_end = b['x']
            y_start, y_end = b['y']
            z_start, z_end = b['z']
            state[:, ch, z_start:z_end, y_start:y_end, x_start:x_end] = 1.0
            buildings.append(b)
        
        return buildings

    def _place_access_points(self, state: torch.Tensor, params: dict, buildings: list) -> list:
        G = self.G
        ch = self.config['ch_access']
        access_points = params.get('access_points', [])
        for ap in access_points:
            x, y, z = ap['x'], ap['y'], ap['z']
            state[:, ch, z:z+2, y:y+2, x:x+2] = 1.0
        return access_points

    def _generate_anchor_zones(self, state: torch.Tensor, params: dict, buildings: list, access_points: list):
        G = self.G
        ch = self.config['ch_anchors']
        sl = self.config['street_levels']
        existing = state[:, self.config['ch_existing'], 0, :, :]
        street_mask = 1.0 - existing
        anchors = torch.zeros(1, 1, G, G, G, device=state.device)

        for ap in access_points:
            if ap.get('type') == 'ground' or ap['z'] < sl:
                x, y = ap['x'], ap['y']
                for z in range(sl):
                    anchors[:, 0, z, max(0,y-2):min(G,y+4), max(0,x-2):min(G,x+4)] = 1.0

        for building in buildings:
            if 'gap_facing_x' in building:
                by_start, by_end = building['y']
                gap_x = building['gap_facing_x']
                is_left = building.get('side') == 'left'
                x_start = gap_x if is_left else gap_x - 1
                x_end = gap_x + 1 if is_left else gap_x
                for z in range(sl):
                    anchors[:, 0, z, by_start:min(by_start+4, by_end), max(0,x_start):min(G,x_end)] = 1.0

        for z in range(sl):
            anchors[:, 0, z, :, :] *= street_mask
        state[:, ch:ch+1, :, :, :] = anchors
