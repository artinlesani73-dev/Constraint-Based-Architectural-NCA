# Z-Taper Removal + Low-Z Gap Suppression + Adaptive Coverage (Analysis)

## 1) Remove Z-Taper
Z-taper is a global, non-physical bias that encourages/penalizes volume solely by height relative to access points. In real architectural context, height distribution should emerge from access + ground + fa?ade + structural constraints, not from a global vertical decay. Therefore, removing z-taper improves realism and avoids training the model to "prefer" arbitrary height bands.

Decision: remove z_taper from corridor target computation.

## 2) Low-Z Mass Suppression in "Street" Region (Corrected)
User clarification: do NOT encourage bulk near buildings at low height. Instead, **discourage mass in street parts** (areas farther from buildings) at low heights (z = 0?5 meters).

This is a different logic than "encourage near buildings". The correct interpretation is:
- At low Z, only apply a *negative* pressure in regions that are far from buildings.
- Near-building regions at low Z are left neutral (neither encouraged nor discouraged).

This requires a distinct mask and a one-sided penalty:
- Define a low-Z mask: z in [0..5] (voxel indices in that band).
- Define a ?far-from-building? mask: legal voxels whose distance to any building fa?ade is greater than a threshold (e.g., > 3?5 voxels).
- Apply a **suppression loss** on structure within (low-Z ? far-from-building) only.

Why this is different:
- ?Encourage near buildings? pushes mass into a zone even if it conflicts with other constraints.
- ?Discourage far from buildings? preserves freedom near buildings but protects the open street/gap. This aligns better with urban design: street-level gaps should stay open, while adjacency to buildings can be used but not forced.

## 3) Adaptive Coverage Target (Best Option)
Problem: fixed coverage (e.g., >70%) can reward filling large legal volumes. In wide gaps, this drives bulkiness and reduces porosity.

Best fix: make coverage target scale with legal/corridor volume. Example rule:
- Compute corridor legal volume (or legal volume) and compare it to a reference volume (e.g., context building volume or total grid volume).
- If legal volume is large, lower coverage target (e.g., 70% ? 40?55%).
- If legal volume is small (tight corridors), keep a higher target (e.g., 70%).

Benefits:
- Prevents ?fill everything? solutions when space is large.
- Keeps connectivity in narrow contexts.
- Matches real-world intent: bigger plazas and corridors should allow lighter, more open structures.

Recommended adaptive strategy:
- Define coverage_target = base_target * clamp(ref_volume / legal_volume, min_ratio, 1.0)
  - base_target: 0.7
  - ref_volume: mean building volume or fixed fraction of grid
  - min_ratio: 0.5 (so target never drops below 35%)

## Summary of Decisions
- Remove z-taper entirely.
- Add low-Z ?far-from-building? suppression (one-sided negative, no positive reward near buildings).
- Replace fixed coverage with adaptive coverage target based on legal/corridor volume.

This combination respects real-world logic, preserves open street space, and reduces the incentive to fill large legal volumes.
