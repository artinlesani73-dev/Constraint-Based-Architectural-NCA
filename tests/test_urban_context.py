"""Representation regressions, deliberately not a model benchmark."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from nca.urban_context import validate,fields,scene_hash,summary
FIXTURE=Path(__file__).resolve().parents[1]/'experiments/scenes/urban_context_v1/block-01.json'
class UrbanContextTests(unittest.TestCase):
    def setUp(self):self.scene=json.loads(FIXTURE.read_text())
    def test_units_axes_interfaces_and_preservation(self):
        original=copy.deepcopy(self.scene);f=fields(self.scene)
        self.assertEqual(f['allowed'].shape,(40,96,96))
        self.assertEqual(summary(self.scene)['extent_xyz_m'],[76.80000000000001,76.80000000000001,32.0])
        self.assertFalse(np.any(f['allowed'] & (f['existing']|f['protected'])))
        for mask in f['interfaces'].values():self.assertTrue(f['allowed'][mask].all())
        self.assertEqual(original,self.scene)
        clone=validate(self.scene);clone['buildings'][0]['id']='changed';self.assertEqual(self.scene,original)
    def test_bad_references_and_face_geometry(self):
        for change in (lambda s:s['interfaces'][0].update(building='missing'),
                       lambda s:s['interfaces'][0]['spans'].update(z=[0,40]),
                       lambda s:s['connections'][0].update(target='missing'),
                       lambda s:s.update(shape_xyz=[True,96,40])):
            s=copy.deepcopy(self.scene);change(s)
            with self.assertRaises(ValueError):validate(s)
    def test_overlap_and_protected_interface(self):
        for key in ('buildings','opportunity_regions'):
            s=copy.deepcopy(self.scene);s[key][1]['bounds']=copy.deepcopy(s[key][0]['bounds'])
            with self.assertRaises(ValueError):validate(s)
        s=copy.deepcopy(self.scene);s['protected_regions'].append(dict(id='blocked-face',bounds=dict(x=[24,25],y=[20,24],z=[8,12])))
        with self.assertRaises(ValueError):validate(s)
    def test_hash_covers_context_and_connections(self):
        h=scene_hash(self.scene);s=copy.deepcopy(self.scene);s['origin_xyz_m'][0]=1
        self.assertNotEqual(h,scene_hash(s))
        s=copy.deepcopy(self.scene);s['connections']=s['connections'][:1]
        self.assertNotEqual(h,scene_hash(s))
if __name__=='__main__':unittest.main()
