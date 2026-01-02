#!/usr/bin/env python3
"""
Fine-tuning script for porous, lightweight structures.

Usage:
    python scripts/finetune_porosity.py --epochs 150 --lr 0.0005

This script fine-tunes the v3.1 MODEL C to produce more porous outputs.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'deploy'))

from model_utils import (
    UrbanPavilionNCA,
    UrbanSceneGenerator,
    compute_corridor_target_v31,
    LocalLegalityLoss
)


# ============ LOSS FUNCTIONS ============

class CorridorCoverageLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(self, structure, corridor_target):
        corridor_mask = corridor_target > 0.5
        corridor_volume = corridor_mask.float().sum() + 1e-8
        covered = (structure * corridor_mask.float()).sum()
        coverage_ratio = covered / corridor_volume
        return F.relu(0.7 - coverage_ratio)


class CorridorSpillLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(self, structure, corridor_target, legality_field):
        outside_corridor = (corridor_target < 0.5).float()
        legal_outside = outside_corridor * legality_field
        spill = (structure * legal_outside).sum()
        total_structure = structure.sum() + 1e-8
        return spill / total_structure


class GroundOpennessLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.street_levels = config['street_levels']

    def forward(self, structure, corridor_target, legality_field):
        ground = structure[:, :self.street_levels]
        corridor_ground = corridor_target[:, :self.street_levels]
        outside_corridor_ground = (corridor_ground < 0.5).float()
        unwanted = ground * outside_corridor_ground
        return unwanted.mean()


class ThicknessLoss(nn.Module):
    def __init__(self, max_thickness: int = 1):
        super().__init__()
        self.max_thickness = max_thickness

    def forward(self, structure):
        binary = (structure > 0.5).float()
        eroded = binary.clone()
        for _ in range(self.max_thickness):
            eroded = -F.max_pool3d(-eroded.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze()
        core_ratio = eroded.sum() / (binary.sum() + 1e-8)
        return core_ratio


class SparsityLossV31(nn.Module):
    def __init__(self, target_ratio: float = 0.06):
        super().__init__()
        self.target = target_ratio

    def forward(self, structure, available):
        available_vol = available.sum() + 1e-8
        filled = structure.sum()
        ratio = filled / available_vol
        excess = F.relu(ratio - self.target)
        return excess ** 2


class DensityPenalty(nn.Module):
    def __init__(self, target: float = 0.04):
        super().__init__()
        self.target = target

    def forward(self, structure):
        density = structure.mean()
        return F.relu(density - self.target)


class TotalVariation3D(nn.Module):
    def forward(self, x):
        dz = torch.abs(x[:, 1:, :, :] - x[:, :-1, :, :]).mean()
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return (dz + dy + dx) / 3


class AccessConnectivityLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(self, state):
        cfg = self.config
        structure = state[:, cfg['ch_structure']]
        access = state[:, cfg['ch_access']]
        dilated = F.max_pool3d(structure.unsqueeze(1), 3, 1, 1).squeeze(1)
        connected = (access * dilated).sum()
        total_access = access.sum() + 1e-8
        return 1.0 - (connected / total_access)


class LoadPathLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(self, state):
        cfg = self.config
        structure = state[:, cfg['ch_structure']]
        B, D, H, W = structure.shape
        ground_support = structure[:, 0:1].clone()
        supported = ground_support
        for z in range(1, D):
            below = F.max_pool3d(supported.unsqueeze(1), (1, 3, 3), 1, (0, 1, 1)).squeeze(1)
            supported = torch.max(supported, torch.min(structure[:, z:z+1], below[:, -1:]))
        unsupported = F.relu(structure - supported.expand_as(structure))
        return unsupported.mean()


class CantileverLoss(nn.Module):
    def forward(self, structure):
        below = F.pad(structure[:, :-1], (0, 0, 0, 0, 1, 0))
        unsupported = F.relu(structure - below)
        return unsupported.mean()


class FacadeContactLoss(nn.Module):
    def __init__(self, config: dict, max_contact_ratio: float = 0.15):
        super().__init__()
        self.config = config
        self.max_ratio = max_contact_ratio

    def forward(self, state):
        cfg = self.config
        structure = state[:, cfg['ch_structure']]
        existing = state[:, cfg['ch_existing']]
        dilated_existing = F.max_pool3d(existing.unsqueeze(1), 3, 1, 1).squeeze(1)
        facade_zone = dilated_existing * (1 - existing)
        contact = (structure * facade_zone).sum()
        total_struct = structure.sum() + 1e-8
        ratio = contact / total_struct
        return F.relu(ratio - self.max_ratio)


class PorosityLoss(nn.Module):
    """Encourage internal voids within the structure."""
    def __init__(self, target_porosity: float = 0.25, erosion_steps: int = 1):
        super().__init__()
        self.target_porosity = target_porosity
        self.erosion_steps = erosion_steps

    def forward(self, structure, corridor):
        binary = (structure > 0.5).float()
        eroded = binary.clone()
        for _ in range(self.erosion_steps):
            eroded = -F.max_pool3d(-eroded.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze(0).squeeze(0)

        interior_mask = eroded > 0.5
        if interior_mask.sum() < 10:
            return torch.tensor(0.0, device=structure.device)

        interior_filled = (structure * interior_mask.float()).sum()
        interior_total = interior_mask.float().sum()
        fill_ratio = interior_filled / (interior_total + 1e-8)
        porosity = 1.0 - fill_ratio
        return F.relu(self.target_porosity - porosity)


class SurfaceAreaLoss(nn.Module):
    """Encourage higher surface-to-volume ratio."""
    def __init__(self, target_ratio: float = 2.5):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, structure):
        binary = (structure > 0.5).float()
        volume = binary.sum() + 1e-8
        padded = F.pad(binary.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), value=0).squeeze()

        surface = 0
        surface += torch.abs(padded[1:, :, :] - padded[:-1, :, :]).sum()
        surface += torch.abs(padded[:, 1:, :] - padded[:, :-1, :]).sum()
        surface += torch.abs(padded[:, :, 1:] - padded[:, :, :-1]).sum()

        ratio = surface / volume
        return F.relu(self.target_ratio - ratio)


# ============ TRAINER ============

class PorosityFineTuner:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        cfg = config

        max_thickness = cfg.get('max_thickness', 1)
        max_facade = cfg.get('max_facade_contact', 0.15)

        self.legality_loss = LocalLegalityLoss(cfg)
        self.coverage_loss = CorridorCoverageLoss(cfg)
        self.spill_loss = CorridorSpillLoss(cfg)
        self.ground_loss = GroundOpennessLoss(cfg)
        self.thickness_loss = ThicknessLoss(max_thickness)
        self.sparsity_loss = SparsityLossV31(target_ratio=0.06)
        self.facade_loss = FacadeContactLoss(cfg, max_facade)
        self.access_conn_loss = AccessConnectivityLoss(cfg)
        self.loadpath_loss = LoadPathLoss(cfg)
        self.cantilever_loss = CantileverLoss()
        self.density_loss = DensityPenalty(target=0.04)
        self.tv_loss = TotalVariation3D()
        self.porosity_loss = PorosityLoss(target_porosity=0.25)
        self.surface_loss = SurfaceAreaLoss(target_ratio=2.5)

        self.weights = {
            'legality': 30.0,
            'coverage': 20.0,
            'spill': 25.0,
            'ground': 35.0,
            'thickness': 50.0,
            'sparsity': 45.0,
            'facade': 10.0,
            'access_conn': 15.0,
            'loadpath': 10.0,
            'cantilever': 5.0,
            'density': 8.0,
            'tv': 0.5,
            'porosity': 20.0,
            'surface': 10.0,
        }

        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr_finetune', 0.0005))
        self.scene_gen = UrbanSceneGenerator(cfg)
        self.history = []

    def _random_scene_params(self):
        G = self.config['grid_size']
        gap_width = np.random.randint(8, 16)
        gap_center = G // 2
        left_x = (0, gap_center - gap_width // 2)
        right_x = (gap_center + gap_width // 2, G)

        buildings = [
            {'x': list(left_x), 'y': [0, np.random.randint(20, 28)], 'z': [0, np.random.randint(10, 18)],
             'side': 'left', 'gap_facing_x': left_x[1]},
            {'x': list(right_x), 'y': [0, np.random.randint(16, 24)], 'z': [0, np.random.randint(10, 18)],
             'side': 'right', 'gap_facing_x': right_x[0]}
        ]

        access_points = [{'x': gap_center, 'y': np.random.randint(18, 26), 'z': 0, 'type': 'ground'}]
        if np.random.random() > 0.3:
            access_points.append({
                'x': np.random.choice([left_x[1], right_x[0]]),
                'y': np.random.randint(2, 8),
                'z': np.random.randint(6, 12),
                'type': 'elevated'
            })

        return {'buildings': buildings, 'access_points': access_points}

    def train_epoch(self, epoch):
        self.model.train()
        cfg = self.config
        batch_size = cfg.get('batch_size', 4)

        seeds, corridors = [], []
        for _ in range(batch_size):
            params = self._random_scene_params()
            seed, _ = self.scene_gen.generate(params, device=self.device)
            corridor = compute_corridor_target_v31(seed, cfg)
            seeds.append(seed)
            corridors.append(corridor)

        seeds = torch.cat(seeds, dim=0)
        corridor_target = torch.cat(corridors, dim=0)

        seed_scale = cfg.get('corridor_seed_scale', 0.15)
        if seed_scale > 0:
            seeds[:, cfg['ch_structure']] = torch.clamp(
                seeds[:, cfg['ch_structure']] + seed_scale * corridor_target, 0, 1
            )

        self.optimizer.zero_grad()
        steps = np.random.randint(40, 60)
        final = self.model(seeds, steps=steps)

        structure = final[:, cfg['ch_structure']]
        existing = final[:, cfg['ch_existing']]
        available = 1.0 - existing
        legality_field = self.legality_loss.compute_legality_field(final)

        L_legality = self.legality_loss(final)
        L_coverage = self.coverage_loss(structure, corridor_target)
        L_spill = self.spill_loss(structure, corridor_target, legality_field)
        L_ground = self.ground_loss(structure, corridor_target, legality_field)
        L_thickness = self.thickness_loss(structure)
        L_sparsity = self.sparsity_loss(structure, available)
        L_facade = self.facade_loss(final)
        L_access = self.access_conn_loss(final)
        L_loadpath = self.loadpath_loss(final)
        L_cant = self.cantilever_loss(structure)
        L_density = self.density_loss(structure)
        L_tv = self.tv_loss(structure)
        L_porosity = self.porosity_loss(structure, corridor_target)
        L_surface = self.surface_loss(structure)

        total_loss = sum(self.weights[k] * v for k, v in [
            ('legality', L_legality), ('coverage', L_coverage), ('spill', L_spill),
            ('ground', L_ground), ('thickness', L_thickness), ('sparsity', L_sparsity),
            ('facade', L_facade), ('access_conn', L_access), ('loadpath', L_loadpath),
            ('cantilever', L_cant), ('density', L_density), ('tv', L_tv),
            ('porosity', L_porosity), ('surface', L_surface)
        ])

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.get('grad_clip', 1.0))
        self.optimizer.step()

        metrics = {
            'epoch': epoch, 'total_loss': total_loss.item(),
            'L_porosity': L_porosity.item(), 'L_thickness': L_thickness.item(),
            'fill_ratio': structure.mean().item()
        }
        self.history.append(metrics)
        return metrics

    def evaluate(self, n_samples=8):
        self.model.eval()
        cfg = self.config
        results = []

        with torch.no_grad():
            for _ in range(n_samples):
                params = self._random_scene_params()
                seed, _ = self.scene_gen.generate(params, device=self.device)
                corridor = compute_corridor_target_v31(seed, cfg)

                seed_scale = cfg.get('corridor_seed_scale', 0.15)
                if seed_scale > 0:
                    seed[:, cfg['ch_structure']] += seed_scale * corridor

                grown = self.model(seed, steps=50)
                structure = grown[:, cfg['ch_structure']]

                corridor_mask = corridor > 0.5
                coverage = (structure * corridor_mask.float()).sum() / (corridor_mask.sum() + 1e-8)
                outside = (corridor < 0.5).float()
                spill = (structure * outside).sum() / (structure.sum() + 1e-8)

                results.append({
                    'coverage': coverage.item(),
                    'spill': spill.item(),
                    'fill_ratio': structure.mean().item()
                })

        return {
            'avg_coverage': np.mean([r['coverage'] for r in results]),
            'avg_spill': np.mean([r['spill'] for r in results]),
            'avg_fill_ratio': np.mean([r['fill_ratio'] for r in results])
        }

    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'weights': self.weights,
            'history': self.history
        }, path)
        print(f'Saved checkpoint to {path}')


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description='Fine-tune NCA for porous structures')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--eval-every', type=int, default=10, help='Evaluate every N epochs')
    parser.add_argument('--save-every', type=int, default=25, help='Save every N epochs')
    parser.add_argument('--output-dir', type=str, default='notebooks/finetuned_porous', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Paths
    base_model_dir = PROJECT_ROOT / 'notebooks' / 'sel' / 'MODEL C - Copy - no change'
    config_path = base_model_dir / 'config_step_b.json'
    checkpoint_path = base_model_dir / 'v31_fixed_geometry.pth'
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['max_thickness'] = 1
    config['lr_finetune'] = args.lr
    config['batch_size'] = args.batch_size

    print(f'Config loaded with modifications:')
    print(f"  max_thickness: {config['max_thickness']}")
    print(f"  lr_finetune: {config['lr_finetune']}")

    # Load model
    model = UrbanPavilionNCA(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print('Model loaded')

    # Create trainer
    trainer = PorosityFineTuner(model, config, device)

    print(f'\nStarting fine-tuning for {args.epochs} epochs...')

    best_fill_ratio = 1.0
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(epoch)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:4d} | Loss: {metrics['total_loss']:.2f} | "
                  f"Fill: {metrics['fill_ratio']*100:.1f}% | "
                  f"Poros: {metrics['L_porosity']:.3f}")

        if epoch % args.eval_every == 0:
            eval_results = trainer.evaluate(n_samples=8)
            print(f"  [EVAL] Coverage: {eval_results['avg_coverage']*100:.1f}% | "
                  f"Spill: {eval_results['avg_spill']*100:.1f}% | "
                  f"Fill: {eval_results['avg_fill_ratio']*100:.1f}%")

            if eval_results['avg_fill_ratio'] < best_fill_ratio and eval_results['avg_coverage'] > 0.5:
                best_fill_ratio = eval_results['avg_fill_ratio']
                trainer.save_checkpoint(output_dir / 'best_porous.pth', epoch)

        if epoch % args.save_every == 0:
            trainer.save_checkpoint(output_dir / f'checkpoint_epoch_{epoch}.pth', epoch)

    trainer.save_checkpoint(output_dir / 'final_porous.pth', args.epochs)
    print('\nFine-tuning complete!')

    # Copy to deploy
    deploy_dir = PROJECT_ROOT / 'deploy' / 'models'
    deploy_dir.mkdir(exist_ok=True)

    import shutil
    best_path = output_dir / 'best_porous.pth'
    if best_path.exists():
        shutil.copy(best_path, deploy_dir / 'v31_porous.pth')
    else:
        shutil.copy(output_dir / 'final_porous.pth', deploy_dir / 'v31_porous.pth')

    with open(deploy_dir / 'config_porous.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f'Model exported to {deploy_dir}')


if __name__ == '__main__':
    main()
