from pathlib import Path
from copy import deepcopy
import json
import os
import subprocess
import sys
import time
import unittest
import uuid
import zipfile
from hashlib import sha256
import numpy as np
from nca.repair_portable import PortableSession,TrainingOrder,read_portable
from nca.repair_training import RepairSession
from nca.recovery import tree_equal
from nca.experiments import digest
from nca.colab_package import safe_name,extract_checked

ROOT=Path(__file__).resolve().parents[1]


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'.local-artifacts/testing/NR2'/uuid.uuid4().hex;self.root.mkdir(parents=True)
        target=np.zeros((6,)*3,np.float32);target[1:5,1:5,1:5]=1
        damaged=target.copy();damaged[2:4,2:4,2:4]=0
        context=np.zeros((7,6,6,6),np.float32);context[:2]=1;context[6]=.24
        self.inputs={'occupancy':damaged,'context':context};self.target=target;self.rows=[]
        for index in range(3):
            path=self.root/f'{index}.npz'
            with path.open('xb') as f:np.savez_compressed(f,target=target,damaged=damaged,condition=context)
            self.rows.append({'arrays':path.name,'arrays_sha256':digest(path),'split':'train'})

    def test_sampler_restores_across_epoch_without_duplicates(self):
        a=TrainingOrder(3,1203)
        first=[a.next() for _ in range(3)];self.assertEqual(sorted(first),[0,1,2])
        b=TrainingOrder(3,0);b.restore(a.state())
        self.assertEqual([a.next() for _ in range(8)],[b.next() for _ in range(8)])
        bad=a.state();bad['position']=100
        with self.assertRaises(ValueError):b.restore(bad)

    def test_cpu_math_matches_nr1_actual_updates(self):
        legacy=RepairSession(self.inputs,self.target,{'unit':'parity'})
        portable=PortableSession(self.root,self.rows[:1],{'unit':'parity'})
        for _ in range(2):
            old,old_state=legacy.step();new,new_state=portable.step()
            self.assertEqual(old['loss'],new['loss']);self.assertEqual(old['pre_clip_gradient_norm'],new['pre_clip_gradient_norm'])
            np.testing.assert_array_equal(old_state,new_state)
            self.assertTrue(tree_equal(legacy.model.state_dict(),portable.model.state_dict()))
            self.assertTrue(tree_equal(legacy.optimizer.state_dict(),portable.optimizer.state_dict()))

    def test_multisample_checkpoint_restores_full_state(self):
        a=PortableSession(self.root,self.rows,{'unit':'resume'});a.step();a.step()
        checkpoint=self.root/'checkpoint.pt';a.save(checkpoint)
        expected=[a.step() for _ in range(3)];state=a.payload()
        b=PortableSession(self.root,self.rows,{'unit':'resume'});b.restore(checkpoint)
        actual=[b.step() for _ in range(3)]
        for x,y in zip(expected,actual):self.assertEqual(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
        self.assertTrue(tree_equal(state,b.payload()))

    def test_heldout_data_and_changed_identity_rejected(self):
        bad=deepcopy(self.rows);bad[0]['split']='test'
        with self.assertRaises(ValueError):PortableSession(self.root,bad,{})
        a=PortableSession(self.root,self.rows,{'unit':'a'});p=self.root/'checkpoint.pt';a.save(p)
        with self.assertRaises(ValueError):PortableSession(self.root,self.rows,{'unit':'b'}).restore(p)
        with self.assertRaises(FileExistsError):a.save(p)

    def test_checksum_and_uncommitted_checkpoint_rejected(self):
        a=PortableSession(self.root,self.rows,{});p=self.root/'checkpoint.pt';a.save(p)
        with p.open('ab') as f:f.write(b'corruption')
        with self.assertRaises(ValueError):read_portable(p,a.identity)
        q=self.root/'uncommitted.pt';q.write_bytes(b'partial')
        with self.assertRaises(FileNotFoundError):read_portable(q,a.identity)

    def test_archive_traversal_rejected_before_extraction(self):
        for name in ('../secret','/root','C:/secret','a\\b','a/../b','a//b'):
            with self.assertRaises(ValueError):safe_name(name)
        p=self.root/'bad.zip'
        with zipfile.ZipFile(p,'x') as z:
            z.writestr('../escape',b'x');z.writestr('manifest.json',json.dumps({'files':{'../escape':sha256(b'x').hexdigest()}}))
        with self.assertRaises(ValueError):extract_checked(p,self.root/'extracted',digest(p))
        self.assertFalse((self.root/'extracted').exists())

    def test_parent_watchdog_allows_native_imports_and_stops_on_eof(self):
        from deploy.studio_process import WorkerTree
        from scripts.colab_repair_preflight import wait_before_deadline
        marker=self.root/'imports-complete.txt'
        code="""import sys,threading,time
from pathlib import Path
from scripts.colab_repair_preflight import watch_parent
assert sys.stdin.readline()=='GO\\n'
threading.Thread(target=watch_parent,daemon=True).start()
import numpy,torch
Path(sys.argv[1]).write_text('ready',encoding='utf-8')
time.sleep(30)
"""
        with (self.root/'watchdog.log').open('x',encoding='utf-8') as log:
            p=subprocess.Popen([sys.executable,'-c',code,str(marker)],cwd=ROOT,
                               stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
            tree=WorkerTree(p)
            try:
                p.stdin.write(b'GO\n');p.stdin.flush()
                deadline=time.monotonic()+20
                while not marker.exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.05)
                self.assertTrue(marker.exists(),'Native imports blocked; inspect retained watchdog.log')
                with self.assertRaises(TimeoutError):wait_before_deadline(p,time.monotonic()+.2)
                p.stdin.close()
                wait_before_deadline(p,time.monotonic()+5)
                self.assertEqual(p.returncode,71)
            finally:
                if tree.active_count():tree.terminate()
                tree.close();p.wait(timeout=5)
                if not p.stdin.closed:p.stdin.close()


if __name__=='__main__':unittest.main()
