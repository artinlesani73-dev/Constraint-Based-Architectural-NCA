# R3 revisit and draft workflow — 2026-10-04

Ready: http://127.0.0.1:8016/ . Select a saved run, then Edit selected run to load its exact scene, volume and seed into the editor. New generations preserve parent_run lineage. Changing to a preset clears the parent link. Outputs and original runs are immutable; edits only affect a new request.

Save draft locally persists a new UUID version on disk, including blank/incomplete numeric fields. Restore draft loads it after a page reload. Draft validation is deferred to generation, which retains the full geometry checks. Saving is explicit, not automatic: save before reloading/switching sites. Draft endpoints inherit loopback Host/Origin/header restrictions and16KiB body limit. All draft versions are retained; no deletion feature. The UI displays saved-draft timestamps explicitly in UTC.

Verified through the browser: loaded parent43182d564aa54abeb7d69c1d59752450, retained facade X7 and connection Z9, saved a blank roof field, reloaded/restored that blank and its validation message, then changed roof to26 and seed to2102, saved a second draft and generated childbbf64c65230c4531bf50a887ade4f59d. Child status:completed. Exact parent request link and edited values verified on disk. All original run files match their prior hashes, new run manifest matches, and frozen source hashes match. No model/training/evaluator changes. This is workflow validation, not a generalization result. Screenshot reviewed; mobile and concurrent multi-tab editing unassessed.

Preservation: prior milestone ../G11-R3-Editor-2026-10-04/RESUME.md remains unchanged at8015. The new studio copies the earlier run so it can be reopened, preserving all bytes. Check verification.json. New runs/drafts after archive creation require a new archive; same-disk ZIP is not off-device backup. Repository synchronization pending. No Drive access, paid training, publication, push or MG7 promotion.

Next: meaningful form alternatives on one fixed custom site, retaining saved lineage and the R3 reference. Avoid conflating changing geometry/seed with an improvement in the scientific method. Start with a bounded, versioned diversity change and compare shape differences and all nine families.

Restart: run this folder's server.py using the project's .venv Python after checking8016 is not already serving. Each job is local CPU and serialized. Restart marks incomplete jobs interrupted; it does not resume mid-rollout. Exact source identity and model are bundled.
