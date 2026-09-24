# Generate and compare building volumes

Open http://127.0.0.1:8001/static/live/index.html while the local server is running.

1. Choose a site. Partial obstruction is a useful starting point.
2. Choose16%,24% or32% of the fixed generation region, then a seed. Start with0,
   1 or2; other nonnegative integer seeds up to2147483647 are exploratory.
3. Click Generate volume. The server runs and saves the result automatically.
   Keep the computer/server running while it works. Closing the page does not
   discard a server job. Stopped attempts remain in history and can be retried.
4. Read the nine checks and requested/actual cell counts. A saved result can fail.
   Use slices/cutaway to inspect it; these change the view, not the evaluation.
5. Generate another request, then choose alternatives A/B and Compare alternatives.
   Cards or View result reopen saved volumes. Controls always define the next
   request; the displayed result has its own site/volume/seed label.
6. Export selected result downloads its geometry, decisions and source evidence.
   Import a building-mass export verifies it by replay before saving. Use mass
   exports here; old scaffold exports belong to the earlier workspace.

These are procedural building volumes, not a trained NCA or finished architecture.
Interior spaces, construction and mechanical safety remain later work. The five
sites are fixed development examples. MS1 does not yet support custom sites or
larger grids. Everything saves locally; there is no automatic Google Drive upload.

Start server from the repository when no other Studio instance owns its store:

```powershell
& '.venv/Scripts/python.exe' -m uvicorn deploy.studio:app --host 127.0.0.1 --port 8001 --no-access-log
```

Use one server worker. Do not delete lock/evidence files to force a second owner.
After source changes, first let active jobs finish or explicitly cancel them,
then restart. Interrupted attempts get a new linked ID when retried.
