"""MS2 queue: reuse owned-worker lifecycle, add launch deadline and replay payloads."""
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
import subprocess
import sys
import time
import zipfile
from deploy.studio import ROOT,identifier,encode
from deploy.studio_jobs import JobManager,TERMINAL,write_once,append_event
from deploy import studio_mass_v2 as mass


class MassV2Jobs(JobManager):
    def __init__(self,*args,deadline_seconds=180,**kwargs):
        self.deadline_seconds=deadline_seconds;self.launched=None
        super().__init__(*args,**kwargs)

    def read(self,job_id):
        result=super().read(job_id)
        request=json.loads((self.directory(job_id)/'request.json').read_bytes())
        result['version']=request['version']
        result['operation']='import' if 'import_sha256' in request else 'generate'
        return result

    def submit_mass(self,settings=None,*,payload=None,parent_job=None):
        with self.lock:
            if sum(j['state'] not in TERMINAL for j in self.list()['jobs'])>=4:raise ValueError('Four pending studies already queued')
            if parent_job and self.read(parent_job)['state'] not in {'failed','cancelled','interrupted'}:
                raise ValueError('Only stopped attempts can be retried')
            if payload is not None:settings,scene,version=mass.prepare_import(payload)
            else:
                settings=mass.checked_request(settings);scene=mass.context_for(settings['scene_case'])['scene'];version=mass.VERSION
            job_id=identifier();directory=self.directory(job_id);directory.mkdir()
            (directory/'events').mkdir();(directory/'progress').mkdir()
            request={'id':job_id,'job_id':job_id,'created_at':datetime.now(timezone.utc).isoformat(),
                'version':version,'kind':'mass_result','scene':scene,'scene_hash':mass.scene_hash(scene),
                'mass_request':settings,'parent_job':parent_job,'provenance':self.provenance,
                'worker_deadline_seconds':self.deadline_seconds}
            if payload is not None:
                request['import_sha256']=sha256(encode(payload)).hexdigest();write_once(directory/'import.json',payload)
            write_once(directory/'request.json',request)
            with zipfile.ZipFile(directory/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
                for name,expected in self.provenance['code_sha256'].items():
                    raw=(ROOT/name).read_bytes()
                    if sha256(raw).hexdigest()!=expected:raise ValueError('Source changed; restart Studio before submitting')
                    archive.writestr(name,raw)
            append_event(directory,'queued',request_sha256=sha256(encode(request)).hexdigest())
            return self.read(job_id)

    def retry(self,job_id):
        with self.lock:
            self.read(job_id)
            p=self.directory(job_id);request=json.loads((p/'request.json').read_bytes())
            payload=None
            if 'import_sha256' in request:
                payload=json.loads((p/'import.json').read_bytes())
                if sha256(encode(payload)).hexdigest()!=request['import_sha256']:raise ValueError('Stored import changed')
            return self.submit_mass(request['mass_request'],payload=payload,parent_job=job_id)

    def finish(self):
        if self.process.poll() is None:return
        directory=self.directory(self.active)
        if self.process.returncode==0 and (directory/'import.json').exists():
            try:
                request=json.loads((directory/'request.json').read_bytes());payload=json.loads((directory/'import.json').read_bytes())
                if sha256(encode(payload)).hexdigest()!=request['import_sha256']:raise ValueError('Import payload changed')
                target=self.store/self.active;target.mkdir(parents=True,exist_ok=True)
                write_once(target/'import.json',payload)
            except Exception as error:
                append_event(directory,'failed',reason=str(error));self.release_process();return
        super().finish()

    def tick(self):
        with self.lock:
            if self.process and self.launched is not None:
                request=json.loads((self.directory(self.active)/'request.json').read_bytes())
                elapsed=time.monotonic()-self.launched
                if elapsed>=request['worker_deadline_seconds']:
                    job_id=self.active
                    if self.process.poll() is None:self.tree.terminate()
                    self.process.wait(timeout=5);self.release_process()
                    append_event(self.directory(job_id),'failed',reason='Worker wall deadline exceeded; partial files retained',elapsed_seconds=elapsed)
            prior=self.active
            if self.command is None:
                # The base manager owns process creation and cancellation. Its
                # command injection is local only, never supplied by an API user.
                queued=sorted((j for j in self.list()['jobs'] if j['state']=='queued'),key=lambda j:(j['created_at'],j['id']))
                if queued:self.command=[sys.executable,'-m','deploy.studio_mass_v2_worker',str(self.directory(queued[0]['id']))]
                try:super().tick()
                finally:self.command=None
            else:super().tick()
            if self.process and self.active!=prior:self.launched=time.monotonic()
