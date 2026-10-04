"""Single-owner durable local queue. Workers are separate, cancellable processes."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Event, RLock, Thread
import json
import logging
import os
import re
import subprocess
import sys
import zipfile

from deploy.studio import ROOT, encode, identifier, read_record, save_record
from deploy.studio_process import WorkerTree

TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}
ID = re.compile(r'\d{8}T\d{6}Z_[0-9a-f]{12}')


def write_once(path, value):
    with path.open('xb') as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())


def history(directory):
    events, previous = [], None
    for index, path in enumerate(sorted((directory / 'events').glob('*.json'))):
        wrapped = json.loads(path.read_bytes())
        event = wrapped['event']
        digest = sha256(encode(event)).hexdigest()
        if (wrapped['sha256'] != digest or event['sequence'] != index
                or event['previous'] != previous or path.stem != f'{index:06d}'):
            raise ValueError('Job history failed integrity verification')
        events.append(event)
        previous = digest
    return events, previous


def append_event(directory, state, **detail):
    events, previous = history(directory)
    event = {'sequence': len(events), 'previous': previous,
             'at': datetime.now(timezone.utc).isoformat(), 'state': state, **detail}
    write_once(directory / 'events' / f'{len(events):06d}.json',
               {'event': event, 'sha256': sha256(encode(event)).hexdigest()})


class JobManager:
    def __init__(self, store, provenance, *, start=True, command=None):
        self.store = Path(store)
        self.root = self.store.parent / (self.store.name + '-jobs')
        self.root.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance
        self.lock, self.stop = RLock(), Event()
        self.active, self.process, self.log = None, None, None
        self.tree = None
        self.command = command  # internal injection for lifecycle tests only
        self.owner = (self.root / 'manager.lock').open('a+b')
        if self.owner.seek(0, 2) == 0:
            self.owner.write(b'0'); self.owner.flush()
        self.owner.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError('Another Studio process owns this job store; use one server worker')
        try:
            self.recover()
        except BaseException:
            self.owner.close()
            raise
        self.thread = Thread(target=self.loop, name='studio-jobs', daemon=True)
        if start:
            self.thread.start()

    def directory(self, job_id):
        if not ID.fullmatch(job_id):
            raise ValueError('Invalid job ID')
        return self.root / job_id

    def read(self, job_id):
        directory = self.directory(job_id)
        request = json.loads((directory / 'request.json').read_bytes())
        events, _ = history(directory)
        if not events:
            raise ValueError('Job submission is incomplete')
        if events[0].get('request_sha256') != sha256(encode(request)).hexdigest():
            raise ValueError('Job request failed integrity verification')
        if request['id'] != job_id:
            raise ValueError('Job request identity mismatch')
        last = events[-1]
        stage = last['state']
        if stage == 'running':
            progress = sorted((directory / 'progress').glob('*.json'))
            if progress:
                try:
                    stage = json.loads(progress[-1].read_bytes())['stage']
                except (ValueError, KeyError):
                    pass  # writer may still be publishing this progress event
        return {'id': job_id, 'scene': request['scene'], 'scene_hash': request['scene_hash'],
                'created_at': request['created_at'], 'parent_job': request.get('parent_job'),
                'state': last['state'], 'stage': stage, 'detail': last,
                'history': events, 'kind': request['kind'], 'mass_request': request.get('mass_request')}

    def list(self):
        with self.lock:
            jobs, issues = [], []
            for directory in sorted(self.root.iterdir(), reverse=True):
                if directory.is_dir() and ID.fullmatch(directory.name):
                    try:
                        jobs.append(self.read(directory.name))
                    except (OSError, ValueError, KeyError, TypeError):
                        issues.append(directory.name)
            return {'jobs': jobs, 'integrity_issues': issues}

    def recover(self):
        # The OS store lock excludes another live manager. Never kill persisted
        # PIDs: a PID can be reused. Worker stdin EOF handles parent death.
        for job in self.list()['jobs']:
            if job['state'] in TERMINAL:
                continue
            try:
                record = read_record(self.store, job['id'])
                request = json.loads((self.directory(job['id']) / 'request.json').read_bytes())
                if (any(record.get(key) != request.get(key) for key in
                        ('id', 'scene', 'scene_hash', 'job_id', 'provenance', 'version', 'kind', 'mass_request'))):
                    raise ValueError('Unrelated result')
            except (OSError, ValueError, KeyError, TypeError):
                append_event(self.directory(job['id']), 'interrupted',
                             reason='Server stopped before terminal acknowledgement; retry creates a linked attempt')
            else:
                append_event(self.directory(job['id']), 'completed', result_id=record['id'],
                             reason='Recovered a fully published result')

    def submit(self, scene, scene_hash, parent_job=None, *, mass_request=None):
        with self.lock:
            if sum(j['state'] not in TERMINAL for j in self.list()['jobs']) >= 4:
                raise ValueError('The local queue is full (four pending studies)')
            if parent_job:
                parent = self.read(parent_job)
                if parent['state'] not in {'failed', 'cancelled', 'interrupted'}:
                    raise ValueError('Only stopped or failed jobs can be retried')
            job_id = identifier()
            directory = self.directory(job_id)
            directory.mkdir()
            (directory / 'events').mkdir()
            (directory / 'progress').mkdir()
            request = {'id': job_id, 'created_at': datetime.now(timezone.utc).isoformat(),
                       'version': 'studio_s2', 'kind': 'result', 'scene': scene,
                       'scene_hash': scene_hash, 'job_id': job_id,
                       'parent_job': parent_job, 'provenance': self.provenance}
            if mass_request is not None:
                from deploy.studio_mass import checked_request, context_for
                settings = checked_request(mass_request)
                if context_for(settings['scene_case'])['scene'] != scene:
                    raise ValueError('Mass request scene mismatch')
                request.update(version='studio_mass_v1', kind='mass_result', mass_request=settings)
            write_once(directory / 'request.json', request)
            # Keep the exact source bytes used for a submitted study, not hashes alone.
            with zipfile.ZipFile(directory / 'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
                for name, expected in self.provenance['code_sha256'].items():
                    data = (ROOT / name).read_bytes()
                    if sha256(data).hexdigest() != expected:
                        raise ValueError('Source changed after server startup; restart Studio before submitting')
                    archive.writestr(name, data)
            append_event(directory, 'queued', request_sha256=sha256(encode(request)).hexdigest())
            return self.read(job_id)

    def retry(self, job_id):
        with self.lock:
            job = self.read(job_id)
            return self.submit(job['scene'], job['scene_hash'], parent_job=job_id, mass_request=job.get('mass_request'))

    def release_process(self):
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=5)
        if self.log:
            self.log.close()
        if self.tree:
            self.tree.close()
        self.tree = None
        self.process, self.log, self.active = None, None, None

    def finish(self):
        directory = self.directory(self.active)
        code = self.process.poll()
        if code is None:
            return
        try:
            if code != 0:
                raise RuntimeError(f'Worker exited with code {code}; see retained worker.log')
            wrapped = json.loads((directory / 'candidate.json').read_bytes())
            record = wrapped['record']
            if wrapped['sha256'] != sha256(encode(record)).hexdigest():
                raise ValueError('Worker candidate failed integrity verification')
            request = json.loads((directory / 'request.json').read_bytes())
            for key in ('id', 'scene', 'scene_hash', 'job_id', 'provenance', 'version', 'kind'):
                if record[key] != request[key]:
                    raise ValueError('Worker candidate does not match its submitted request')
            if record.get('mass_request') != request.get('mass_request'):
                raise ValueError('Worker changed the mass parameters')
            save_record(self.store, record)
            append_event(directory, 'completed', result_id=record['id'])
        except Exception as error:
            logging.getLogger(__name__).exception('Could not finalize Studio job %s', self.active)
            append_event(directory, 'failed', reason=str(error))
        finally:
            self.release_process()

    def tick(self):
        with self.lock:
            if self.process:
                self.finish()
            if self.process or self.stop.is_set():
                return
            queued = sorted((j for j in self.list()['jobs'] if j['state'] == 'queued'),
                            key=lambda j: (j['created_at'], j['id']))
            if not queued:
                return
            job_id = queued[0]['id']
            directory = self.directory(job_id)
            try:
                self.log = (directory / 'worker.log').open('xb')
                command = self.command or [sys.executable, '-m', 'deploy.studio_worker', str(directory)]
                flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                    stdout=self.log, stderr=subprocess.STDOUT, creationflags=flags)
                self.tree = WorkerTree(self.process)
                self.active = job_id
                append_event(directory, 'running', worker_pid=self.process.pid)
                # Child must not compute before the running event is durable.
                self.process.stdin.write(b'GO\n'); self.process.stdin.flush()
            except Exception as error:
                if self.process and self.process.poll() is None:
                    self.process.kill()
                append_event(directory, 'failed', reason=str(error))
                self.release_process()

    def cancel(self, job_id, state='cancelled'):
        with self.lock:
            job = self.read(job_id)
            if job['state'] in TERMINAL:
                return job
            if self.active == job_id:
                # Completion wins only once the worker has exited and its result
                # has been validated/published. Otherwise stop actual compute.
                if self.process.poll() is not None:
                    self.finish()
                    return self.read(job_id)
                self.tree.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait(timeout=5)
                pid = self.process.pid
                self.release_process()
                append_event(self.directory(job_id), state, worker_pid=pid,
                             reason='Worker process stopped; partial files retained')
            else:
                append_event(self.directory(job_id), state, reason='Stopped before worker launch')
            return self.read(job_id)

    def loop(self):
        while not self.stop.wait(.2):
            try:
                self.tick()
            except Exception:
                logging.getLogger(__name__).exception('Studio queue error; inspect retained job files')

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=10)
        with self.lock:
            for job in self.list()['jobs']:
                if job['state'] not in TERMINAL:
                    self.cancel(job['id'], state='interrupted')
            self.owner.close()
