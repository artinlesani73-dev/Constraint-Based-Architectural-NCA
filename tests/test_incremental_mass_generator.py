import unittest
import numpy as np
from nca.coverage_mass_generator import grow_coverage as reference,coverage_parts
from nca.incremental_mass_generator import grow_coverage as optimized,IncrementalCubeCounts


class IncrementalTests(unittest.TestCase):
    def test_cache_matches_direct_voxels_many_updates(self):
        rng=np.random.default_rng(817)
        for width in (1,2,3,6):
            domain=rng.random((8,9,10))>.2;parts,_=coverage_parts(domain,.08)
            contact=rng.random(domain.shape)>.7;field=rng.random(domain.shape)>.8
            cache=IncrementalCubeCounts(field,contact,width,parts)
            cells=np.argwhere(~field);rng.shuffle(cells)
            for batch in np.array_split(cells,9):
                cache.remove_cells(batch);field[tuple(batch.T)]=True
                direct=lambda a:np.lib.stride_tricks.sliding_window_view(a,(width,)*3).sum(axis=(-3,-2,-1)).ravel()
                expected=np.stack([direct((~field)&p) for p in parts])
                np.testing.assert_array_equal(cache.by_third,expected)
                np.testing.assert_array_equal(cache.added,expected.sum(0))
                np.testing.assert_array_equal(cache.new_contact,direct((~field)&contact))
            cache.remove_cells(np.empty((0,3),int));self.assertFalse(cache.added.any())

    def compare(self,field,contact,width,target,valid,route,radial,limit,expired,parts,minima):
        first=field.copy();second=field.copy()
        a=reference(first,contact,width,target,valid,route,radial,limit,expired,parts,minima)
        b=optimized(second,contact,width,target,valid,route,radial,limit,expired,parts,minima)
        self.assertEqual(a,b);np.testing.assert_array_equal(first,second)

    def test_overlapping_growth_exact_report(self):
        rng=np.random.default_rng(99);domain=np.ones((5,6,8),bool);parts,minima=coverage_parts(domain,.08)
        for width in (1,2,3):
            field=np.zeros_like(domain);field[:width,:width,:width]=True
            valid=np.ones(tuple(n-width+1 for n in domain.shape),bool)
            for limit in (.15,.4):
                contact=rng.random(domain.shape)>.85;contact[:width,:width,:width]=False
                self.compare(field,contact,width,95,valid,[(0,0,0)],rng.random(valid.shape),limit,lambda:False,parts,minima)

    def test_transit_stall_timeout_exact(self):
        domain=np.ones((1,1,4),bool);parts,minima=coverage_parts(domain,.08)
        for stop in (False,True):
            for contact in (np.zeros_like(domain),np.array([[[False,False,True,True]]])):
                self.compare(np.array([[[True,True,False,False]]]),contact,1,4,domain,[(0,0,0)],np.zeros(domain.shape),.15,lambda:stop,parts,minima)

    def test_request_stop_exact(self):
        domain=np.ones((1,1,3),bool);parts,_=coverage_parts(domain,.08)
        self.compare(np.array([[[True,False,False]]]),np.zeros_like(domain),1,2,domain,[(0,0,0)],np.zeros(domain.shape),.15,lambda:False,parts,np.ones(3,int))


if __name__=='__main__':unittest.main()
