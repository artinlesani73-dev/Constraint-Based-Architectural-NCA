# Run in the same Colab notebook. Change PART to retry/download another part.
from pathlib import Path
import hashlib, math
from google.colab import files

PART = 1
SIZE = 4 * 1024 * 1024
archive = Path(PACKAGE) / 'curriculum-runs' / '20261003T102352Z_a40c9f968e66.zip'
if not archive.is_file():
    raise FileNotFoundError('Original ZIP is missing from this runtime. Do not rerun training; report this error.')
with archive.open('rb') as f:
    actual = hashlib.file_digest(f, 'sha256').hexdigest()
if actual != 'aa842375705080e46833722951c38a893b2a8a2c10594a53426544faba7267b7':
    raise ValueError('Archive differs from the downloaded receipt. Stop and report this.')
total = math.ceil(archive.stat().st_size / SIZE)
if not 1 <= PART <= total:
    raise ValueError(f'PART must be between 1 and {total}')
chunk = archive.with_name(archive.name + f'.part{PART:03d}')
with archive.open('rb') as f:
    f.seek((PART - 1) * SIZE)
    data = f.read(SIZE)
if chunk.exists():
    if chunk.read_bytes() != data:
        raise ValueError('Existing part differs. Stop and report this.')
else:
    with chunk.open('xb') as f:
        f.write(data)
print(f'Downloading part {PART} of {total} ({len(data):,} bytes). Keep the original ZIP.')
print('Part SHA256:', hashlib.sha256(data).hexdigest())
files.download(str(chunk))
