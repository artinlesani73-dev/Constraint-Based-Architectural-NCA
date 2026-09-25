# Building mass Studio, version 2

Open `http://127.0.0.1:8001/static/live-v2/index.html` after starting the local
Studio. The Original Studio link keeps the earlier workflow available.

1. Choose a site. The label includes its grid size. The physical-scale note
   shows the world span: 25.6 m, 38.4 m or 51.2 m. All use 0.8 m voxels;
   the larger grids represent larger sites, not finer resolution.
2. Choose an available volume and variation seed. The five 32-grid presets
   offer 16%, 24% and 32%, with seeds 0–2. The six larger presets offer the
   evaluated 24% setting and seeds 6–7. Original-version records with other
   seeds remain readable and replayable through import.
3. Select Generate volume. A local worker grows and checks the volume, then
   saves the completed result. Cancel stops its process and retains the attempt.
   Retry creates a linked attempt with the same settings. A worker that exceeds
   its wall deadline is stopped while the server is active; suspension can delay
   enforcement. Closing a browser tab does not cancel a server job.
4. Inspect the whole volume, vertical or horizontal slices, and cutaway. The
   cutaway hides the foreground half for viewing; it never changes evaluation.
   A slice index is a voxel layer, and its cell-center height/position is shown.
5. Read all nine checks and the actual/requested voxel count. A blocked preset
   deliberately demonstrates a failed result. Results are retained either way.
6. Compare saved alternatives. Voxel differences are calculated only for the
   same physical scene and generation region. Across different sites, compare
   volume and checks descriptively. Original and incremental versions are labeled.
7. Export a selected result to retain geometry, decisions and original source
   data. Import queues a cancellable replay using the recognized local generator
   version. A verified import creates a new saved attempt; repeated imports do
   not replace earlier records. Source bundles are treated as data, never run.

This is procedural generation, not output from a newly trained NCA. Occupied
voxels represent overall building volume. Interiors, use, construction and
structural analysis are later work. A passing geometric result does not establish
those qualities. The larger presets are a small evaluated set, not evidence of
reliability on arbitrary sites.

Results live locally in `.local-artifacts/studio-mass-v2` and its sibling job
directory. Older results remain in `studio-mass`. Stop/restart recovery preserves
terminal results and marks unfinished attempts interrupted. Use the project
resume document to continue development. Google Drive is not used automatically.
