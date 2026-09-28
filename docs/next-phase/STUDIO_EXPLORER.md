# Studio volume explorer

## Studio volume explorer - 2026-09-28

Implemented /static/live-v3/index.html, linked from live-v2. This is a frontend
upgrade over the existing /api/mass-v2 service: MG7 remains procedural, NR3/4/5
remain experimental. No model, objective, nine-family evaluator, dataset,
preset geometry, generation budget or stored experiment was changed.

Features: orbit by pointer drag or arrow keys; bounded zoom via buttons, +/- or
Shift-wheel; reset view; boundary-face cache and camera-facing culling; world-size
filter for the EXISTING 11 presets at32/48/64; labeled PNG export with record and
method identity; retained orthogonal slices, fixed-Y cutaway, history, replay
import/export and linked job lifecycle. Shared camera aligns comparison views.
Transparent context uses approximate painter ordering; it is illustrative.
No measured speedup claim, new scale benchmark or learned-model scaling claim.

Verified in the in-app browser against saved64-cubed record
20260925T092140Z_b57cc8b2676f: scale64 filter exposes three sites; record loads
4209.15m3 with original passing checks; pointer rotation, zoom/reset, both slices,
cutaway toggle and different-site comparison work; no browser error logs.
PNG saved to Downloads and copied into Codex outputs/Studio-Explorer-2026-09-28;
881x660 PNG signature, all chunk CRCs and decompressed pixel stream verified.
Screenshot retained there as explorer.png. Existing history was read, not rerun.
Generation/import lifecycle code is carried forward; no new generation job or
import replay was run for this presentation-only change. Responsive rules added,
but small-device interaction has not been independently verified.

Recovery notes: server was stopped; restarted local-only uvicorn deploy.studio:app
on127.0.0.1:8001 (PID6576, tool session7847). First preview hit connection refused;
that stale error tab blocked navigation, so a fresh tab was used. Browser download
wait timed out after successful file creation; filesystem verification confirmed
the PNG. PIL unavailable in project venv; used standard-library PNG validation.

Next: expand environment diversity deliberately using the same nine families,
then evaluate those new settings in one bounded local batch before exposing them
as evaluated choices. Existing32/48/64 support is not newly achieved here. Decide
on a unified landing page once the explorer interaction is reviewed; do not
silently replace historical interfaces. No automatic new paid training.
No Drive operation, push or publication. Local archive is same-disk only.
