# R3 comparison preview — 2026-10-04

Separate local saved-output gallery at http://127.0.0.1:8013/ . Open index.html directly if the local server is stopped; data.js is bundled and no network dependencies are used.

## Delivered
All 81 evaluated cases, four exact saved fields each (G10/R3 at 64/128). Shared orthographic camera with drag rotation, zoom, front/plan views and reset. Source colouring distinguishes planner admissions, neural-selected admissions under guards and initial seed. Existing buildings are wireframe overlays, not occluding context. No smoothing, filling, new inference or training.

Each panel retains all nine family pass/fail labels, gross volume, occupied-domain fraction and R3 planner birth share excluding the seed. Frozen evaluation caveats and hybrid identity are visible. These controls explore saved results only; this is not an arbitrary-scene generator or deployment promotion.

## Verification
Export verified source file hashes against the independent-review manifest, all 324 occupancy counts, all R3 planner/learned birth counts and the single initial seed. Browser checked initial rendering, horizon switch, provenance toggle, front/plan/reset and case selection. Confirmed a raw G10 access failure and a regression coverage failure are displayed. No captured browser errors. Screenshot inspected at normal desktop-panel size; mobile and performance benchmarking remain unassessed. Camera drag/zoom are implemented but not automated-tested this turn.

## Decision and next step
Review the paired volumes and planner overlay. Next implementation milestone: package the frozen R3 adapter behind a separate local generation endpoint, with explicit certificate failure handling and per-request provenance/results persistence. Keep raw G10 comparison available. Before broad deployment, assess more geometry families and firing seeds; larger-grid cost remains unmeasured. No automatic paid training, publication or live-model replacement.

## Resume and preservation
Previous milestone: ../G11-R3-Independent-Review-2026-10-04/RESUME.json . This folder is the current preview milestone. All original scientific evidence remains immutable there. This preview has its own hash-verified archive. MG7 unchanged. Repository synchronization remains pending (repository docs still contain older milestones). No Drive operation occurred; same-disk archive is not off-device backup.

Restart: project .venv Python -m http.server 8013 --bind 127.0.0.1 --directory C:/Users/artin/Documents/Codex/outputs/G11-R3-Preview-2026-10-04 . Server logs are operational only and excluded from archive.
