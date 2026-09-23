"""Studio integration checks; these are not learned-model benchmarks."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import torch
from fastapi.testclient import TestClient
from deploy import studio
from nca.contract import load_reference_set
from nca.evaluation import endpoint_connectivity
import numpy as np


class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = load_reference_set()

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.client = TestClient(studio.app)
        self.client.__enter__()
        studio.app.state.store = Path(self.temp.name)
        self.scene = deepcopy(self.scenes['ref-01-ground-pair'])

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_six_presets_and_local_assets(self):
        self.assertEqual(len(self.client.get('/api/studio/scenes').json()['scenes']), 6)
        for path in ('/', '/static/js/studio.js', '/static/css/studio.css'):
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_all_reference_scenes_preserve_feasibility_and_families(self):
        for name, scene in self.scenes.items():
            with self.subTest(scene=name):
                response = self.client.post('/api/studio/records', json={'scene': scene})
                self.assertEqual(response.status_code, 200, response.text)
                record = response.json()
                d = record['diagnostics']
                self.assertFalse(record['learned'])
                self.assertEqual(len(d['families']), 9)
                if name == 'ref-05-sealed-partition':
                    self.assertFalse(d['route_feasible'])
                    self.assertFalse(d['joint_budget_connectivity'])
                else:
                    self.assertTrue(d['joint_budget_connectivity'])
                self.assertEqual(d['material_voxels'], len(record['material_zyx']))
                self.assertEqual(d['legality']['illegal_voxels'], 0)
                self.assertEqual(self.client.get('/api/studio/records/'+record['id']).json(), record)

    def test_deterministic_geometry_without_global_rng_mutation(self):
        before = torch.get_rng_state().clone()
        a = studio.plan_scene(self.scene, studio.app.state.config)
        b = studio.plan_scene(self.scene, studio.app.state.config)
        self.assertEqual(a['material_zyx'], b['material_zyx'])
        self.assertEqual(a['diagnostics'], b['diagnostics'])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_edit_changes_scene_identity_and_records_never_overwrite(self):
        a = self.client.post('/api/studio/records?plan=false', json={'scene': self.scene}).json()
        self.scene['buildings'][0]['z'][1] -= 1
        b = self.client.post('/api/studio/records?plan=false', json={'scene': self.scene}).json()
        self.assertNotEqual(a['scene_hash'], b['scene_hash'])
        self.assertNotEqual(a['id'], b['id'])
        self.assertEqual(self.client.get('/api/studio/records/'+a['id']).json(), a)
        with self.assertRaises(FileExistsError):
            studio.save_record(studio.app.state.store, a)

    def test_reject_out_of_bounds_and_buried_entrances(self):
        for x in (32, 1):
            self.scene['entrances'][0]['x'] = x
            self.assertEqual(self.client.post('/api/studio/records', json={'scene': self.scene}).status_code, 422)
        self.assertEqual(self.client.get('/api/studio/records').json()['records'], [])

    def test_resource_and_contract_bounds(self):
        for key,value in [('grid_size',10000),('street_levels',2),('voxel_size_m',1),('ceiling_z',25),('entrances',[]),('legacy_relaxations',['facade_below_street_band'])]:
            scene = {**self.scene, key:value}
            self.assertEqual(self.client.post('/api/studio/validate', json={'scene':scene}).status_code,422)
        self.scene['buildings'][0]['gap_facing_x'] = 15
        self.assertEqual(self.client.post('/api/studio/validate',json={'scene':self.scene}).status_code,422)

    def test_corruption_and_incomplete_records_are_visible(self):
        record=self.client.post('/api/studio/records?plan=false',json={'scene':self.scene}).json()
        path=studio.app.state.store/record['id']/'record.json'
        path.write_text('{}')
        self.assertEqual(self.client.get('/api/studio/records/'+record['id']).status_code,409)
        incomplete=studio.identifier()
        (studio.app.state.store/incomplete).mkdir()
        listed=self.client.get('/api/studio/records').json()
        self.assertEqual(listed['records'],[])
        self.assertEqual(set(listed['integrity_issues']),{record['id'],incomplete})

    def test_failed_plans_are_retained(self):
        with patch.object(studio,'plan_scene',side_effect=RuntimeError('injected failure')):
            self.assertEqual(self.client.post('/api/studio/records',json={'scene':self.scene}).status_code,500)
        records=self.client.get('/api/studio/records').json()['records']
        self.assertEqual(records[0]['kind'],'failure')
        failure=self.client.get('/api/studio/records/'+records[0]['id']).json()
        self.assertEqual(failure['error'],'injected failure')
        submitted=studio.app.state.store/failure['failed_attempt_id']/'request.json'
        self.assertEqual(json.loads(submitted.read_bytes())['scene'],self.scene)

    def test_duplicate_worker_rejected_and_lock_released_on_error(self):
        studio.GATE.acquire()
        try:
            self.assertEqual(self.client.post('/api/studio/records',json={'scene':self.scene}).status_code,429)
        finally:
            studio.GATE.release()
        self.test_failed_plans_are_retained()
        self.assertFalse(studio.GATE.locked())

    def test_cross_origin_and_large_payload_rejected(self):
        self.assertEqual(self.client.post('/api/studio/records',json={'scene':self.scene},headers={'Origin':'https://elsewhere.example'}).status_code,403)
        self.assertEqual(self.client.post('/api/studio/validate',content=b'x'*100001).status_code,413)

    def test_saved_record_survives_a_new_service_session(self):
        a=self.client.post('/api/studio/records?plan=false',json={'scene':self.scene}).json()
        with TestClient(studio.app) as restarted:
            studio.app.state.store=Path(self.temp.name)
            self.assertEqual(restarted.get('/api/studio/records/'+a['id']).json(),a)

    def test_record_identifier_cannot_escape_archive(self):
        for value in ('../outside','bad','20260924T000000Z_ffffffffffff/../../x'):
            with self.assertRaises(ValueError):
                studio.read_record(studio.app.state.store,value)


if __name__ == '__main__':
    unittest.main()
