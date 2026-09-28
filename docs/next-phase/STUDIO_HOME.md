# Unified Studio home

## Unified Studio home - 2026-09-28

Root / now serves deploy/static/home/index.html with Explore, Research and Archive
navigation. /scaffold serves the original deploy/studio.html byte-for-byte.
Existing live/live-v2/live-v3 explicit scaffold links point there; brand links
return to the home. Generator APIs, geometry, models and stored records unchanged.
No new generation or training run. No push, publishing or Drive access.

Homepage clearly separates procedural MG7 live generation from experimental NCA;
shows dated15-site/three-scale scope, NR3/NR4/NR5 all-nine counts25/19/17 out of27,
ED1 eight-of-eight pilot result and limitations. Downloadable JSON copies match
experiments/reports/NR5-single-trial-review.json and ED1-diversity.json exactly.
These are static dated research snapshots; update them deliberately with future
findings. Original research galleries and historical semantics are preserved.

Checked all12 local href targets with HTTP200, original scaffold response against
source bytes, both research copies against their originals. Browser screenshot
and Research anchor checked; no console errors. Responsive CSS included; no
independent mobile device verification. This completes the current local entry/
navigation batch, not public-hosting readiness or learned-model admission.

Screenshot and verified source/doc archive: Codex outputs/Studio-Home-2026-09-28.
Server PID10740/session96370; local-only127.0.0.1:8001. Restart was performed after
confirming all three job queues had no queued/running tasks. Resume command:
.venv/Scripts/python.exe -m uvicorn deploy.studio:app --host127.0.0.1 --port8001 --no-access-log
(with spaces between option names and values). Same-disk archive, not off-device.

Next: review the consolidated product experience with the user; select the next
substantial research or deployment objective. No automatic additional seed/loss
sweeps or paid training. Current live NCA reliability limitation remains D084.
