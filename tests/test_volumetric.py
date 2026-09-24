import unittest
import numpy as np
from nca.volumetric import measure_volume,bracket_axes,free_cube_centers,fixed_probes


class VolumeTests(unittest.TestCase):
    def field(self):return np.zeros((9,9,9),bool)
    def report(self,m,e=None,region=None):return measure_volume(m,np.zeros_like(m) if e is None else e,np.ones_like(m) if region is None else region)[0]
    def shell(self):
        m=self.field();m[2:7,2:7,2:7]=True;m[3:6,3:6,3:6]=False;return m

    def test_analytic_closed_shell(self):
        r=self.report(self.shell())
        self.assertEqual(r['material_voxels'],98);self.assertEqual(r['sealed_by_form_voxels'],27)
        self.assertEqual(r['bracketed_3_axes_voxels'],27)
        self.assertEqual(r['free_cube_centers_in_bracketed_void']['3'],1)

    def test_open_shell_has_space_without_sealed_cavity(self):
        m=self.shell();m[2,3:6,3:6]=False;r=self.report(m)
        self.assertEqual(r['sealed_by_form_voxels'],0)
        self.assertGreater(r['bracketed_2_axes_voxels'],0)
        self.assertEqual(r['bracketed_exterior_connected_voxels'],r['bracketed_2_axes_voxels'])

    def test_context_closure_reported_separately(self):
        m=self.shell();m[2,3:6,3:6]=False;e=self.field();e[2,3:6,3:6]=True;r=self.report(m,e)
        self.assertEqual(r['sealed_by_form_voxels'],0);self.assertEqual(r['sealed_with_context_voxels'],27)

    def test_solid_block_extent_does_not_imply_void(self):
        m=self.field();m[2:7,2:7,2:7]=True;r=self.report(m)
        self.assertEqual(r['extent_balance'],1);self.assertEqual(r['bracketed_2_axes_voxels'],0)

    def test_empty_and_plane_have_no_bracketed_space(self):
        m=self.field();r=self.report(m);self.assertIsNone(r['bbox_zyx']);self.assertIsNone(r['extent_balance'])
        m[4,:,:]=True;r=self.report(m)
        self.assertEqual(r['sealed_by_form_voxels'],0);self.assertEqual(r['bracketed_2_axes_voxels'],0)

    def test_region_boundary_does_not_seal_air(self):
        m=self.field();m[:,2,:]=True;region=self.field();region[3:6,3:6,3:6]=True
        self.assertEqual(self.report(m,region=region)['sealed_with_context_voxels'],0)

    def test_axis_permutation_preserves_counts(self):
        m=self.shell();m[2,3:6,3:6]=False
        a=self.report(m);b=self.report(m.transpose(2,0,1))
        for key in ('bracketed_2_axes_voxels','bracketed_3_axes_voxels','sealed_by_form_voxels','material_voxels'):
            self.assertEqual(a[key],b[key])

    def test_clearance_does_not_wrap_or_extend_outside_grid(self):
        free=np.ones((5,5,5),bool)
        self.assertEqual(int(free_cube_centers(free,3).sum()),27)
        self.assertEqual(int(free_cube_centers(free,5).sum()),1)
        self.assertEqual(int(free_cube_centers(free,7).sum()),0)

    def test_extent_decoy_has_eight_components_and_no_void(self):
        m=fixed_probes()['extent_decoy_8'];r=self.report(m)
        self.assertEqual(r['material_components_6'],8);self.assertEqual(r['extent_cells_zyx'],[9,9,16])
        self.assertEqual(r['bracketed_2_axes_voxels'],0)

    def test_fixed_mass_shapes_and_closed_shell_analytic_volume(self):
        p=fixed_probes()
        for key in ('slab_512','solid_512','open_ends_512'):self.assertEqual(int(p[key].sum()),512)
        self.assertEqual(int(p['side_aperture_487'].sum()),487)
        self.assertEqual(self.report(p['closed_shell_610'])['sealed_by_form_voxels'],14*7*7)

    def test_existing_context_cannot_supply_brackets(self):
        m=self.field();e=self.shell();r=self.report(m,e)
        self.assertEqual(r['bracketed_2_axes_voxels'],0);self.assertEqual(r['sealed_with_context_voxels'],27)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):self.report(np.zeros((9,9,9)))
        with self.assertRaises(ValueError):self.report(self.field(),region=self.field())
        with self.assertRaises(ValueError):free_cube_centers(self.field(),2)
        with self.assertRaises(ValueError):measure_volume(self.field(),self.field(),~self.field(),True)


if __name__=='__main__':unittest.main()
