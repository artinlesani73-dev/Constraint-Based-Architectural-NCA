# NR4 preservation trial - PREPARED ONLY

One proposed fresh seed1201,256 updates,16 rollout steps per update,600s job cap.
No GPU run is authorized yet. Keep APPROVED_SEED_JOB=False until this exact job
is approved. Do not run the whole notebook while it is disarmed.

Use a Tesla T4 Colab runtime with the previously verified software stack.
Open NCA-NR4-Preservation.ipynb locally in Colab and upload ONLY
NCA-NR4-Preservation-Package.zip in the upload cell. After compute approval,
set APPROVED_SEED_JOB=True and execute the training cell once. Download both
result ZIP and receipt even after failure. Return them for local verification.
Disconnect the idle GPU after downloads are verified. No automatic retry or
second seed. Do not delete attempt markers to bypass a failed job.

The job timer excludes setup/export/idle allocation and does not cap billing.
Whole-VM loss before download can lose this one job. Checkpoints bind to the
runtime; they do not guarantee exact recovery in a new VM. No Drive mounting,
uploading, reading, or notebook autosave is authorized by this package. Each
Drive action needs separate approval within the project folder.

Only the reconstruction objective changes: base balanced BCE +0.5 negative-class
BCE +1.0 base BCE on intact examples. No extra constraint family, architecture,
training distribution, optimizer, grid, horizon or postprocessing change.
This balances preservation against repair; stronger penalties may suppress repair.
The training labels determine intact status; no intact flag/target enters inference.

81 TRAIN rows only. After this proposed single run, review checkpoint256 on the
same27 VALIDATION examples at32 steps/firing2101 on CPU. Compare against NR3,
unchanged and closing3. No TEST scoring, checkpoint selection, threshold search,
extra model seeds or automatic promotion. Validation is now development evidence,
not an untouched test. Loss values across NR3/NR4 are not directly comparable.
