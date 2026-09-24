import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from deploy.checkpoints import load_model_c
from nca.massing_cases import target_scenes, target_context, target_controls
from nca.massing_objective import make_context
from nca.massing_direct import MassLogits, MassingSession
from nca.recovery import tree_equal
from nca.mass_generator import MassGeneratorSpec, generate_mass
from nca.contact_mass_generator import ContactGeneratorSpec, generate_contact_mass


class MassingDirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = dict(target_scenes())['aligned']
        cls.fields, cls.domain, _ = target_context(cls.scene, load_model_c(device='cpu')[0])
        cls.context = make_context(cls.scene, cls.fields, cls.domain)
        cls.initial = target_controls(cls.scene, cls.domain)['compact_mass']
        cls.recipe = json.loads((Path(__file__).resolve().parents[1] / 'experiments/configs/MD1-direct.json').read_bytes())

    def session(self, identity=None):
        return MassingSession(self.initial, self.context, self.recipe, identity or {'fixture': 'test'})

    def test_iteration_zero_and_projection_under_extreme_logits(self):
        model = MassLogits(self.initial, self.domain)
        np.testing.assert_array_equal(model().detach().numpy() > .5, self.initial)
        for value in (-1000., 1000.):
            with torch.no_grad(): model.logits.fill_(value)
            self.assertEqual(model()[~model.domain].sum().item(), 0.)

    def test_outside_domain_initial_is_rejected(self):
        with self.assertRaises(ValueError): MassLogits(np.ones_like(self.domain), self.domain)

    def test_actual_update_checkpoint_replay_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'state.pt'
            session = self.session(); evidence = session.step()
            self.assertTrue(evidence['gradient_finite']); self.assertGreater(evidence['gradient_norm'], 0)
            session.save(path)
            with self.assertRaises(FileExistsError): session.save(path)
            session.step(); final, p = session.evaluate()
            restored = self.session(); self.assertEqual(restored.restore(path), 1)
            restored.step(); replay, q = restored.evaluate()
            self.assertEqual(final, replay); np.testing.assert_array_equal(p, q)
            self.assertTrue(tree_equal(session.optimizer.state_dict(), restored.optimizer.state_dict()))

    def test_checkpoint_rejects_different_source_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'state.pt'; self.session().save(path)
            with self.assertRaises(ValueError): self.session({'source': 'different'}).restore(path)

    def test_zero_contact_cost_preserves_original_geometry(self):
        a, route, report = generate_mass(self.scene, self.fields, self.domain, 0, MassGeneratorSpec())
        b, other_route, new = generate_contact_mass(self.scene, self.fields, self.domain, 0, ContactGeneratorSpec(contact_weight=0))
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(route, other_route)
        self.assertEqual(report['selected_origins_zyx'], new['selected_origins_zyx'])

    def test_contact_control_uses_same_domain_and_is_deterministic(self):
        domain = self.domain.copy()
        a, route, report = generate_contact_mass(self.scene, self.fields, domain, 1)
        b, other_route, new = generate_contact_mass(self.scene, self.fields, domain, 1)
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(route, other_route)
        np.testing.assert_array_equal(domain, self.domain)
        self.assertFalse((a & ~domain).any())
        self.assertEqual(report['selected_origins_zyx'], new['selected_origins_zyx'])
        self.assertGreater(report['spec']['contact_weight'], 0)


if __name__ == '__main__': unittest.main()
