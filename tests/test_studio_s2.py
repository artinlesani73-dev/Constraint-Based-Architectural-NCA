"""Actual process lifecycle, recovery, portable integrity and revision comparisons."""
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
from deploy import studio
from deploy.studio_jobs import JobManager, append_event, history, write_once, TERMINAL
from deploy.studio_process import WorkerTree
from deploy.studio_portable import import_verified, compare


def until(predicate, timeout=20):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=predicate()
        if value:return value
        time.sleep(.05)
    raise AssertionError('Timed out waiting for lifecycle state')


class S2Tests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.patch=patch.object(studio,'STORE',self.root/'records');self.patch.start()
        self.client=TestClient(studio.app);self.client.__enter__()
        self.scene=self.client.get('/api/studio/scenes').json()['scenes'][0]
        self.manager=studio.app.state.jobs

    def tearDown(self):
        self.client.__exit__(None,None,None);self.patch.stop();self.temp.cleanup()

    def submit(self):
        r=self.client.post('/api/studio/jobs',json={'scene':self.scene})
        self.assertEqual(r.status_code,202,r.text)
        return r.json()

    def terminal(self,job):
        return until(lambda:(v if (v:=self.manager.read(job['id']))['state'] in TERMINAL else None))

    def result(self):
        return self.client.post('/api/studio/records',json={'scene':self.scene}).json()

    def test_real_worker_completes_and_saves_exact_procedural_result(self):
        job=self.submit();done=self.terminal(job)
        self.assertEqual(done['state'],'completed',done)
        record=studio.read_record(self.manager.store,job['id'])
        control=studio.plan_scene(self.scene,studio.app.state.config)
        self.assertEqual(record['diagnostics'],control['diagnostics'])
        self.assertEqual(record['material_zyx'],control['material_zyx'])
        self.assertTrue((self.manager.directory(job['id'])/'source.zip').exists())
        self.assertEqual([e['state'] for e in done['history']],['queued','running','completed'])
        self.assertEqual(self.manager.cancel(job['id'])['state'],'completed')

    def test_cancel_stops_actual_worker_and_retry_links_attempt(self):
        job=self.submit()
        until(lambda:self.manager.read(job['id'])['state']=='running')
        process=self.manager.process
        self.assertIsNone(process.poll())
        response=self.client.post('/api/studio/jobs/'+job['id']+'/cancel')
        self.assertEqual(response.json()['state'],'cancelled')
        self.assertIsNotNone(process.poll())
        self.assertFalse((self.manager.store/job['id']/'receipt.json').exists())
        retry=self.client.post('/api/studio/jobs/'+job['id']+'/retry').json()
        self.assertNotEqual(retry['id'],job['id'])
        self.assertEqual(retry['parent_job'],job['id'])
        self.assertEqual(self.terminal(retry)['state'],'completed')
        self.assertEqual(self.manager.read(job['id'])['state'],'cancelled')

    def test_cancel_queued_job_and_queue_bound(self):
        isolated=JobManager(self.root/'isolated',studio.app.state.provenance,start=False)
        try:
            jobs=[isolated.submit(self.scene,studio.scene_hash(self.scene)) for _ in range(4)]
            with self.assertRaises(ValueError):isolated.submit(self.scene,studio.scene_hash(self.scene))
            self.assertEqual(isolated.cancel(jobs[0]['id'])['state'],'cancelled')
            self.assertIsNone(isolated.process)
        finally:isolated.close()

    def test_second_manager_cannot_own_same_store(self):
        with self.assertRaises(RuntimeError):JobManager(self.manager.store,studio.app.state.provenance,start=False)

    def test_recovery_marks_unacknowledged_job_interrupted(self):
        isolated=JobManager(self.root/'isolated',studio.app.state.provenance,start=False)
        job=isolated.submit(self.scene,studio.scene_hash(self.scene))
        append_event(isolated.directory(job['id']),'running',worker_pid=999999)
        isolated.owner.close()  # simulate OS releasing ownership after a crash
        recovered=JobManager(self.root/'isolated',studio.app.state.provenance,start=False)
        try:
            self.assertEqual(recovered.read(job['id'])['state'],'interrupted')
            self.assertEqual(recovered.retry(job['id'])['parent_job'],job['id'])
        finally:recovered.close()

    def test_recovery_recognizes_published_result_before_final_event(self):
        isolated=JobManager(self.root/'isolated',studio.app.state.provenance,start=False)
        job=isolated.submit(self.scene,studio.scene_hash(self.scene))
        request=json.loads((isolated.directory(job['id'])/'request.json').read_bytes())
        record={**request,**studio.plan_scene(self.scene,studio.app.state.config)}
        studio.save_record(isolated.store,record)
        isolated.owner.close()
        recovered=JobManager(isolated.store,studio.app.state.provenance,start=False)
        try:self.assertEqual(recovered.read(job['id'])['state'],'completed')
        finally:recovered.close()

    def test_failed_worker_and_tampered_history_remain_visible(self):
        isolated=JobManager(self.root/'isolated',studio.app.state.provenance,start=False,
                            command=[sys.executable,'-c','import sys;sys.stdin.buffer.readline();sys.exit(7)'])
        try:
            job=isolated.submit(self.scene,studio.scene_hash(self.scene));isolated.tick()
            until(lambda:isolated.process.poll() is not None);isolated.tick()
            self.assertEqual(isolated.read(job['id'])['state'],'failed')
            event=isolated.directory(job['id'])/'events/000000.json'
            value=json.loads(event.read_bytes());value['event']['state']='completed'
            event.write_text(json.dumps(value))
            self.assertEqual(isolated.list()['integrity_issues'],[job['id']])
        finally:isolated.close()

    def test_checksum_semantics_and_duplicate_import(self):
        record=self.result()
        payload=self.client.get('/api/studio/records/'+record['id']+'/export').json()
        self.assertTrue(self.client.post('/api/studio/import',json=payload).json()['duplicate'])
        foreign=deepcopy(payload);foreign['record']['id']=studio.identifier()
        foreign['sha256']=sha256(studio.encode(foreign['record'])).hexdigest()
        imported=self.client.post('/api/studio/import',json=foreign)
        self.assertEqual(imported.status_code,200,imported.text)
        self.assertFalse(imported.json()['duplicate'])
        self.assertNotEqual(imported.json()['record']['id'],foreign['record']['id'])
        self.assertTrue(self.client.post('/api/studio/import',json=foreign).json()['duplicate'])
        foreign['record']['diagnostics']['material_ratio']=.09
        self.assertEqual(self.client.post('/api/studio/import',json=foreign).status_code,422)
        foreign['sha256']=sha256(studio.encode(foreign['record'])).hexdigest()
        self.assertEqual(self.client.post('/api/studio/import',json=foreign).status_code,422)

    def test_forged_geometry_rejected_even_with_recomputed_checksum(self):
        record=self.result();record['material_zyx'][0]=[99,1,1]
        payload={'format':'studio_portable_v1','record':record,'sha256':sha256(studio.encode(record)).hexdigest()}
        self.assertEqual(self.client.post('/api/studio/import',json=payload).status_code,422)

    def test_legacy_import_requires_exact_local_record(self):
        record=self.result()
        self.assertTrue(self.client.post('/api/studio/import',json=record).json()['duplicate'])
        record['id']=studio.identifier()
        self.assertEqual(self.client.post('/api/studio/import',json=record).status_code,422)

    def test_revision_comparison_reports_changes_and_exact_geometry_delta(self):
        a=self.result();self.scene['buildings'][0]['z'][1]-=1;b=self.result()
        value=self.client.get('/api/studio/compare',params={'a':a['id'],'b':b['id']}).json()
        self.assertFalse(value['same_scene'])
        self.assertEqual(value['scene_changes'][0]['id'],'B_west')
        self.assertEqual(value['added_voxels'],0)
        self.assertEqual(value['shared_voxels'],len(a['material_zyx']))
        c=deepcopy(a);c['material_zyx']=a['material_zyx'][1:]+[[31,31,31]]
        delta=compare(a,c)
        self.assertEqual((delta['added_voxels'],delta['removed_voxels']),(1,1))

    def test_shutdown_interrupts_live_worker(self):
        job=self.submit();until(lambda:self.manager.read(job['id'])['state']=='running')
        process=self.manager.process
        self.manager.close()
        self.assertIsNotNone(process.poll())
        self.assertEqual(self.manager.read(job['id'])['state'],'interrupted')

    def test_windows_owner_death_stops_worker_and_grandchild(self):
        if os.name != 'nt':
            # POSIX support uses the worker pipe watchdog, not Windows jobs.
            return
        import ctypes
        from ctypes import wintypes
        marker=self.root/'descendants.json'
        child_code="import sys,subprocess,time,json,os;from pathlib import Path;sys.stdin.buffer.readline();p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);Path(sys.argv[1]).write_text(json.dumps([os.getpid(),p.pid]));time.sleep(60)"
        host_code="import sys,subprocess,time;from deploy.studio_process import WorkerTree;p=subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1]],stdin=subprocess.PIPE);tree=WorkerTree(p);p.stdin.write(b'GO\\n');p.stdin.flush();time.sleep(60)"
        flags=subprocess.CREATE_NO_WINDOW
        host=subprocess.Popen([sys.executable,'-c',host_code,str(marker),child_code],cwd=studio.ROOT,creationflags=flags)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handles=[]
        try:
            until(marker.exists)
            pids=json.loads(marker.read_text())
            handles=[kernel.OpenProcess(0x100000,False,pid) for pid in pids]
            self.assertTrue(all(handles))
            for handle in handles:self.assertEqual(kernel.WaitForSingleObject(handle,0),258)
            host.kill();host.wait(timeout=5)
            for handle in handles:self.assertEqual(kernel.WaitForSingleObject(handle,5000),0)
        finally:
            if host.poll() is None:host.kill();host.wait(timeout=5)
            for handle in handles:
                if handle:kernel.CloseHandle(handle)


if __name__=='__main__':unittest.main()
