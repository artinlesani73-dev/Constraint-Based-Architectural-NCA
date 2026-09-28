"""MS2 scale/version boundaries and real queue/replay lifecycle checks."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import subprocess
import sys
import time
import unittest
from fastapi.testclient import TestClient
from deploy import studio,studio_mass_v2 as mass,studio_mass as legacy
from deploy.studio_mass_v2_jobs import MassV2Jobs
from deploy.studio_jobs import TERMINAL,append_event


def until(fn,seconds=45):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        value=fn()
        if value:return value
        time.sleep(.05)
    raise AssertionError('MS2 test deadline reached')


class MassV2Tests(unittest.TestCase):
    def setUp(self):
        self.root=studio.ROOT/'.local-artifacts/testing/MS2'/studio.identifier();self.root.mkdir(parents=True)
        self.patch=patch.object(studio,'STORE',self.root/'records');self.patch.start()
        self.client=TestClient(studio.app);self.client.__enter__();self.m=studio.app.state.mass_v2_jobs
        self.settings={'scene_case':'partial_obstruction','seed':2,'request_fraction':.24}

    def tearDown(self):
        self.client.__exit__(None,None,None);self.patch.stop()

    def done(self,job):
        value=until(lambda:(v if (v:=self.m.read(job['id']))['state'] in TERMINAL else None))
        self.assertEqual(value['state'],'completed',value)
        return self.client.get('/api/mass-v2/records/'+job['id']).json()

    def submit(self,settings=None):
        response=self.client.post('/api/mass-v2/jobs',json=settings or self.settings)
        self.assertEqual(response.status_code,202,response.text);return response.json()

    def exported(self):
        value=self.done(self.submit());return self.client.get('/api/mass-v2/records/'+value['id']+'/export').json()

    def test_presets_and_bounded_coordinates(self):
        data=self.client.get('/api/mass-v2/presets').json();self.assertEqual(len(data['contexts']),15)
        self.assertEqual(sum(len(c['seeds'])*len(c['requests']) for c in data['contexts']),65)
        for value in ({'scene_case':'64__compact','seed':0,'request_fraction':.24},
                      {'scene_case':'48__compact','seed':6,'request_fraction':.32},
                      {**self.settings,'seed':True},{**self.settings,'seed':3}):
            self.assertEqual(self.client.post('/api/mass-v2/jobs',json=value).status_code,422)
        for coords in ([[64,0,0]],[[1.0,0,0]],[[True,0,0]],[[0,0,0],[0,0,0]]):
            with self.assertRaises(ValueError):mass.grid(coords,64)
        self.assertTrue(mass.grid([[63,63,63]],64)[63,63,63])
        for c in data['contexts']:
            if c['case'].startswith('ed1_'):
                self.assertEqual(c['seeds'],[6,7]);self.assertEqual(c['requests'],[.24])
                domain,fields=mass.arrays(mass.context_for(c['case']))
                self.assertEqual(domain.shape,(48,48,48))
                self.assertFalse((domain & fields['existing']).any())
        self.assertEqual(self.client.post('/api/mass-v2/import',content=b' '*20_000_001,headers={'content-type':'application/json'}).status_code,413)
        self.assertEqual(self.client.post('/api/mass-v2/jobs',json=self.settings,headers={'origin':'https://unrelated.example'}).status_code,403)

    def test_real_workers_all_sizes_and_blocked(self):
        for case,seed in [('partial_obstruction',2),('48__offset_obstacle',6),('64__offset_obstacle',7),('64__blocked',6)]:
            settings={**self.settings,'scene_case':case,'seed':seed};value=self.done(self.submit(settings))
            computed=mass.generate(settings)
            self.assertEqual(value['occupied_zyx'],computed['occupied_zyx']);self.assertEqual(value['targets'],computed['targets'])
            self.assertEqual(value['method'],mass.GENERATOR)
            self.assertEqual(value['targets']['contract_pass'],case!='64__blocked')

    def test_new_import_foreign_reexport_and_tampering(self):
        payload=self.exported();payload['record']['id']=studio.identifier();payload['sha256']=sha256(mass.encode(payload['record'])).hexdigest()
        response=self.client.post('/api/mass-v2/import',json=payload);self.assertEqual(response.status_code,202,response.text)
        result=self.done(response.json());out=self.client.get('/api/mass-v2/records/'+result['id']+'/export').json()
        self.assertEqual(out['source_files'],payload['source_files']);self.assertEqual(out['source_provenance'],payload['source_provenance'])
        again=self.client.post('/api/mass-v2/import',json=out);self.assertEqual(again.status_code,202,again.text);self.done(again.json())
        for field in ('occupied_zyx','generation'):
            bad=deepcopy(payload)
            if field=='occupied_zyx':bad['record'][field].pop()
            else:bad['record'][field]['growth']['trace'][0]['radial_score']+=1
            bad['sha256']=sha256(mass.encode(bad['record'])).hexdigest()
            queued=self.client.post('/api/mass-v2/import',json=bad);self.assertEqual(queued.status_code,202,queued.text)
            job=queued.json();done=until(lambda:(v if (v:=self.m.read(job['id']))['state'] in TERMINAL else None))
            self.assertEqual(done['state'],'failed');self.assertFalse((self.m.store/job['id']/'receipt.json').exists())
        for field in ('source','version'):
            bad=deepcopy(payload)
            if field=='source':bad['source_files']['nca/incremental_mass_generator.py']+='\n# forged'
            else:bad['record']['version']='unknown';bad['sha256']=sha256(mass.encode(bad['record'])).hexdigest()
            self.assertEqual(self.client.post('/api/mass-v2/import',json=bad).status_code,422)

    def test_legacy_import_and_cross_version_compare(self):
        oldmanager=studio.app.state.mass_jobs
        scene=legacy.context_for(self.settings['scene_case'])['scene']
        job=oldmanager.submit(scene,studio.scene_hash(scene),mass_request=self.settings)
        until(lambda:oldmanager.read(job['id'])['state'] in TERMINAL)
        self.assertEqual(oldmanager.read(job['id'])['state'],'completed')
        payload=self.client.get('/api/mass/records/'+job['id']+'/export').json()
        self.assertEqual(self.client.get('/api/mass-v2/records/'+job['id']).status_code,200)
        queued=self.client.post('/api/mass-v2/import',json=payload);self.assertEqual(queued.status_code,202,queued.text)
        imported=self.done(queued.json());self.assertEqual(imported['version'],legacy.VERSION)
        out=self.client.get('/api/mass-v2/records/'+imported['id']+'/export').json()
        self.assertEqual(out['format'],legacy.FORMAT);self.assertEqual(out['source_files'],payload['source_files'])
        new=self.done(self.submit());comparison=self.client.get('/api/mass-v2/compare',params={'a':job['id'],'b':new['id']}).json()
        self.assertTrue(comparison['same_context'])
        other=self.done(self.submit({'scene_case':'48__compact','request_fraction':.24,'seed':6}))
        comparison=self.client.get('/api/mass-v2/compare',params={'a':job['id'],'b':other['id']}).json()
        self.assertFalse(comparison['same_context']);self.assertIsNone(comparison['added_voxels'])

    def test_cancel_queue_running_and_linked_retry(self):
        first=self.submit({'scene_case':'64__offset_obstacle','seed':7,'request_fraction':.24})
        until(lambda:self.m.read(first['id'])['state']=='running');process=self.m.process
        second=self.submit();self.m.cancel(second['id']);self.assertEqual(self.m.read(second['id'])['state'],'cancelled')
        self.m.cancel(first['id']);self.assertEqual(self.m.read(first['id'])['state'],'cancelled');self.assertIsNotNone(process.poll())
        retry=self.m.retry(first['id']);self.assertEqual(retry['parent_job'],first['id']);self.done(retry)

    def test_deadline_and_recovery(self):
        command=[sys.executable,'-c','import sys,time; sys.stdin.buffer.readline(); time.sleep(60)']
        isolated=MassV2Jobs(self.root/'deadline',studio.app.state.provenance,start=False,deadline_seconds=.1,command=command)
        try:
            job=isolated.submit_mass(self.settings);isolated.tick();process=isolated.process;time.sleep(.15);isolated.tick()
            self.assertEqual(isolated.read(job['id'])['state'],'failed');self.assertIsNotNone(process.poll())
            self.assertIn('deadline',isolated.read(job['id'])['detail']['reason'])
        finally:isolated.close()
        isolated=MassV2Jobs(self.root/'recovery',studio.app.state.provenance,start=False)
        job=isolated.submit_mass(self.settings);isolated.owner.close()
        recovered=MassV2Jobs(isolated.store,studio.app.state.provenance,start=False)
        try:
            self.assertEqual(recovered.read(job['id'])['state'],'interrupted')
            retry=recovered.retry(job['id']);self.assertEqual(retry['mass_request'],self.settings)
        finally:recovered.close()

    def test_matching_published_result_recovers(self):
        isolated=MassV2Jobs(self.root/'published',studio.app.state.provenance,start=False)
        job=isolated.submit_mass(self.settings);request=json.loads((isolated.directory(job['id'])/'request.json').read_bytes())
        studio.save_record(isolated.store,{**request,**mass.generate(self.settings)});isolated.owner.close()
        recovered=MassV2Jobs(isolated.store,studio.app.state.provenance,start=False)
        try:self.assertEqual(recovered.read(job['id'])['state'],'completed')
        finally:recovered.close()

    def test_parent_death_stops_real_ms2_worker(self):
        if os.name!='nt':return
        import ctypes
        from ctypes import wintypes
        provenance=self.root/'parent-provenance.json';provenance.write_bytes(studio.encode(studio.app.state.provenance))
        marker=self.root/'parent-worker.json';store=self.root/'parent-store'
        host_code='''import json,sys,time
from pathlib import Path
from deploy.studio_mass_v2_jobs import MassV2Jobs
m=MassV2Jobs(Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_bytes()))
j=m.submit_mass({'scene_case':'64__offset_obstacle','seed':7,'request_fraction':.24})
end=time.monotonic()+15
while m.process is None and time.monotonic()<end:time.sleep(.01)
assert m.process is not None
Path(sys.argv[3]).write_text(json.dumps({'job':j['id'],'pid':m.process.pid}))
time.sleep(60)
'''
        host=subprocess.Popen([sys.executable,'-c',host_code,str(store),str(provenance),str(marker)],cwd=studio.ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=None
        try:
            until(marker.exists);data=json.loads(marker.read_bytes());handle=kernel.OpenProcess(0x100000,False,data['pid'])
            self.assertTrue(handle);self.assertEqual(kernel.WaitForSingleObject(handle,0),258)
            host.kill();host.wait(timeout=5);self.assertEqual(kernel.WaitForSingleObject(handle,5000),0)
            recovered=MassV2Jobs(store,studio.app.state.provenance,start=False)
            try:self.assertEqual(recovered.read(data['job'])['state'],'interrupted')
            finally:recovered.close()
        finally:
            if host.poll() is None:host.kill();host.wait(timeout=5)
            if handle:kernel.CloseHandle(handle)


if __name__=='__main__':unittest.main()
