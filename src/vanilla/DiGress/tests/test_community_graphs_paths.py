"""CommunityGraphs path resolution (Docker vs OGAD monorepo)."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from src.datasets.community_graphs_paths import resolve_community_graphs_pickle

REL = '../../../data/train/graphs/CommunityGraphs.pickle'


class TestResolveCommunityGraphsPickle(unittest.TestCase):
    def test_docker_flat_layout(self):
        """Only ``<root>/data/train/graphs/...`` exists (e.g. image WORKDIR /app + volume)."""
        base = tempfile.mkdtemp()
        self.addCleanup(lambda b=base: __import__('shutil').rmtree(b, ignore_errors=True))
        pick = os.path.join(base, 'data', 'train', 'graphs', 'CommunityGraphs.pickle')
        os.makedirs(os.path.dirname(pick), exist_ok=True)
        open(pick, 'wb').close()
        got = resolve_community_graphs_pickle(base, REL)
        self.assertTrue(os.path.samefile(got, pick))

    def test_monorepo_layout(self):
        """Primary path under DiGress root exists (three levels up to OGAD ``data``)."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda b=tmp: __import__('shutil').rmtree(b, ignore_errors=True))
        # Simulate OGAD/src/vanilla/DiGress so ../../../data/... stays inside tmp (sandbox-safe).
        fake_ogad = os.path.join(tmp, 'ogad_root')
        base = os.path.join(fake_ogad, 'src', 'vanilla', 'DiGress')
        primary = os.path.normpath(os.path.join(base, REL))
        os.makedirs(os.path.dirname(primary), exist_ok=True)
        open(primary, 'wb').close()
        got = resolve_community_graphs_pickle(base, REL)
        self.assertTrue(os.path.samefile(got, primary))

    def test_missing_raises(self):
        base = tempfile.mkdtemp()
        self.addCleanup(lambda b=base: __import__('shutil').rmtree(b, ignore_errors=True))
        with self.assertRaises(FileNotFoundError):
            resolve_community_graphs_pickle(base, REL)


@unittest.skipUnless(
    importlib.util.find_spec('hydra') is not None,
    'hydra-core not installed',
)
class TestHydraCommunityGraphsExperiment(unittest.TestCase):
    def test_compose_selects_dataset(self):
        from hydra import compose, initialize_config_dir

        digress_root = Path(__file__).resolve().parents[1]
        config_dir = digress_root / 'configs'
        with initialize_config_dir(config_dir=str(config_dir), version_base='1.3'):
            cfg = compose(
                config_name='config',
                overrides=['+experiment=community_graphs'],
            )
        self.assertEqual(cfg.dataset.name, 'community_graphs')


if __name__ == '__main__':
    unittest.main()
