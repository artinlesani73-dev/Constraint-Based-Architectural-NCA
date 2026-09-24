"""Real mass jobs, typed recovery, benchmark parity and portable semantic integrity."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import time
import unittest
import numpy as np
from fastapi.testclient import TestClient
from deploy import studio
from deploy.studio_mass import generate, contexts, encode, FORMAT
from deploy.studio_jobs import JobManager, TERMINAL, append_event
from nca.massing_cases import target_context


def until(predicate, timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        result=predicate()
        if result:return result
        time.sleep(.05)
    raise AssertionError('Mass worker timed out')


class MassStudioTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.root=Path(self.temp.name)
        self.patch=patch.object(studio,'STORE',self.root/'records');self.patch.start()
        self.client=TestClient(studio.app);self.client.__enter__()
        self.manager=studio.app.state.mass_jobs
        self.settings={'scene_case':'partial_obstruction','request_fraction':.32,'seed':2}

    def tearDown(self):
        self.client.__exit__(None,None,None);self.patch.stop();self.temp.cleanup()

    def job(self):
        result=self.client.post('/api/mass/jobs',json=self.settings)
        self.assertEqual(result.status_code,202,result.text)
        return result.json()

    def completed(self):
        job=self.job()
        value=until(lambda:(v if (v:=self.manager.read(job['id']))['state'] in TERMINAL else None))
        self.assertEqual(value['state'],'completed',value)
        result=self.client.get('/api/mass/records/'+job['id'])
        self.assertEqual(result.status_code,200,result.text)
        return result.json()

    def test_frozen_inputs_match_live_context_builder(self):
        for context in contexts():
            fields,domain,_=target_context(context['scene'],studio.app.state.config)
            self.assertEqual(np.argwhere(domain).tolist(),context['domain_zyx'])
            for key,value in context['masks'].items():self.assertEqual(np.argwhere(fields[key]).tolist(),value)

    def test_real_worker_matches_saved_MG3_and_exports_exact_source(self):
        result=self.completed()
        data=json.loads((studio.ROOT/'deploy/static/budget/study.json').read_bytes())
        expected=next(c for c in data['cases'] if c['case']=='partial_obstruction__v32__s2')
        for key in ('occupied_zyx','targets','field_sha256'):self.assertEqual(result[key],expected[key])
        self.assertEqual(result['mass_request'],self.settings)
        self.assertEqual(len(result['generation']['growth']['trace']),result['generation']['growth']['evaluated_proposals'])
        payload=self.client.get('/api/mass/records/'+result['id']+'/export').json()
        self.assertEqual(payload['format'],FORMAT)
        for name,digest in result['provenance']['code_sha256'].items():
            self.assertEqual(sha256(payload['source_files'][name].encode()).hexdigest(),digest)
        self.assertEqual(self.client.get('/api/studio/records/'+result['id']).status_code,404)
        self.assertEqual(self.client.get('/api/studio/records').json()['records'],[])

    def test_blocked_result_is_completed_job_with_failed_science(self):
        self.settings['scene_case']='blocked_gap'
        result=self.completed()
        self.assertFalse(result['targets']['contract_pass'])
        self.assertEqual(result['generation']['status'],'no_cube_route')
        self.assertEqual(result['occupied_zyx'],[])
        self.assertEqual(len(result['targets']['family_pass']),9)

    def test_request_bounds_and_old_type_rejection(self):
        for bad in ({**self.settings,'seed':True},{**self.settings,'seed':-1},
                    {**self.settings,'seed':2**31},{**self.settings,'seed':2.0},
                    {**self.settings,'request_fraction':.25},{**self.settings,'scene_case':'custom'},
                    {**self.settings,'scene':{}},{'scene':{}}):
            self.assertEqual(self.client.post('/api/mass/jobs',json=bad).status_code,422,bad)
        self.assertEqual(self.client.post('/api/mass/import',json={'format':'studio_portable_v1'}).status_code,422)
        self.assertEqual(self.client.post('/api/studio/import',json={'format':FORMAT}).status_code,422)

    def test_active_cancel_retry_preserves_every_parameter(self):
        job=self.job();until(lambda:self.manager.read(job['id'])['state']=='running')
        process=self.manager.process
        response=self.client.post('/api/mass/jobs/'+job['id']+'/cancel')
        self.assertEqual(response.json()['state'],'cancelled')
        self.assertIsNotNone(process.poll())
        retry=self.client.post('/api/mass/jobs/'+job['id']+'/retry').json()
        self.assertEqual(retry['parent_job'],job['id']);self.assertEqual(retry['mass_request'],self.settings)
        done=until(lambda:(v if (v:=self.manager.read(retry['id']))['state'] in TERMINAL else None))
        self.assertEqual(done['state'],'completed',done)
        self.assertEqual(self.client.get('/api/mass/records/'+retry['id']).json()['mass_request'],self.settings)

    def test_recovery_retains_request_and_rejects_mismatched_published_result(self):
        isolated=JobManager(self.root/'isolated',studio.app.state.provenance,start=False)
        scene=next(c['scene'] for c in contexts() if c['case']==self.settings['scene_case'])
        job=isolated.submit(scene,studio.scene_hash(scene),mass_request=self.settings)
        request=json.loads((isolated.directory(job['id'])/'request.json').read_bytes())
        result={**request,**generate({**self.settings,'seed':1})}
        studio.save_record(isolated.store,result)
        append_event(isolated.directory(job['id']),'running',worker_pid=999999)
        isolated.owner.close()
        recovered=JobManager(isolated.store,studio.app.state.provenance,start=False)
        try:
            self.assertEqual(recovered.read(job['id'])['state'],'interrupted')
            retry=recovered.retry(job['id']);self.assertEqual(retry['mass_request'],self.settings)
        finally:recovered.close()

    def test_restart_recognizes_matching_published_mass_record(self):
        isolated=JobManager(self.root/'isolated',studio.app.state.provenance,start=False)
        scene=next(c['scene'] for c in contexts() if c['case']==self.settings['scene_case'])
        job=isolated.submit(scene,studio.scene_hash(scene),mass_request=self.settings)
        request=json.loads((isolated.directory(job['id'])/'request.json').read_bytes())
        studio.save_record(isolated.store,{**request,**generate(self.settings)})
        isolated.owner.close()
        recovered=JobManager(isolated.store,studio.app.state.provenance,start=False)
        try:self.assertEqual(recovered.read(job['id'])['state'],'completed')
        finally:recovered.close()

    def test_import_replays_trace_and_rejects_forged_geometry_and_sources(self):
        result=self.completed();payload=self.client.get('/api/mass/records/'+result['id']+'/export').json()
        response=self.client.post('/api/mass/import',json=payload)
        self.assertEqual(response.status_code,200,response.text);self.assertTrue(response.json()['duplicate'])
        forged=deepcopy(payload);forged['record']['occupied_zyx'].pop()
        forged['sha256']=sha256(encode(forged['record'])).hexdigest()
        self.assertEqual(self.client.post('/api/mass/import',json=forged).status_code,422)
        forged=deepcopy(payload);forged['record']['generation']['growth']['trace'][0]['accepted']=False
        forged['sha256']=sha256(encode(forged['record'])).hexdigest()
        self.assertEqual(self.client.post('/api/mass/import',json=forged).status_code,422)
        forged=deepcopy(payload);forged['source_files']['nca/budget_mass_generator.py']+='\n# altered'
        self.assertEqual(self.client.post('/api/mass/import',json=forged).status_code,422)

    def test_foreign_import_reexport_and_duplicate_without_overwrite(self):
        result=self.completed();payload=self.client.get('/api/mass/records/'+result['id']+'/export').json()
        payload['record']['id']=studio.identifier();payload['sha256']=sha256(encode(payload['record'])).hexdigest()
        response=self.client.post('/api/mass/import',json=payload)
        self.assertEqual(response.status_code,200,response.text);value=response.json()
        self.assertFalse(value['duplicate']);self.assertNotEqual(value['record']['id'],payload['record']['id'])
        second=self.client.post('/api/mass/import',json=payload).json();self.assertTrue(second['duplicate'])
        reexport=self.client.get('/api/mass/records/'+value['record']['id']+'/export')
        self.assertEqual(reexport.status_code,200,reexport.text)
        self.assertEqual(reexport.json()['source_files'],payload['source_files'])
        self.assertEqual(self.client.post('/api/mass/import',json=reexport.json()).status_code,200)
        self.assertEqual(len(self.client.get('/api/mass/records').json()['records']),2)

    def test_compare_uses_mass_metrics_and_actual_voxel_difference(self):
        first=self.completed();self.settings['request_fraction']=.16;second=self.completed()
        response=self.client.get('/api/mass/compare',params={'a':first['id'],'b':second['id']})
        self.assertEqual(response.status_code,200,response.text);comparison=response.json()
        a=set(map(tuple,first['occupied_zyx']));b=set(map(tuple,second['occupied_zyx']))
        self.assertEqual(comparison['added_voxels'],len(b-a));self.assertEqual(comparison['removed_voxels'],len(a-b))
        self.assertTrue(comparison['same_context'])


if __name__=='__main__':unittest.main()
