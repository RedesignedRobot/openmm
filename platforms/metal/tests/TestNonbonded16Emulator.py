"""CPU rules for the Stage 1 packet census. Pass the emulator directory as argv[1]."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(sys.argv.pop(1)).resolve()))
from emulate import analyze, half_distances


class TestNonbonded16Emulator(unittest.TestCase):
    def test_receiver_selection_matches_periodic_pair_oracle(self):
        rng = np.random.default_rng(2026)
        # Skew box, coordinates outside the primary image, and a partial x block.
        box = np.array([[4., 0, 0], [1.9, 4., 0], [-1.8, 1.7, 4.]])
        positions = (rng.random((77, 3))*3-1) @ box
        atoms = np.array([list(range(32, 64)), list(range(32))])
        atoms[1, -1] = 96
        tiles = np.array([0, 2])
        actual = half_distances(positions, tiles, atoms, 77, box) < 0.9**2
        shifts = np.array([(i, j, k) for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)]) @ box
        expected = np.zeros((2, 2, 32), dtype=bool)
        for t, x in enumerate(tiles):
            for half in range(2):
                for lane, y in enumerate(atoms[t]):
                    if y >= 77:
                        continue
                    for a in range(x*32+half*16, min(x*32+(half+1)*16, 77)):
                        d = positions[a]-positions[y]-shifts
                        expected[t, half, lane] |= np.any(np.sum(d*d, axis=1) < 0.9**2)
        np.testing.assert_array_equal(actual, expected)

    def test_compaction_crosses_tiles_but_never_x_halves(self):
        # Each half needs sixteen receivers from each old tile. Packing across
        # tiles gives two packets total, while retaining tile boundaries gives four.
        posq = np.zeros((96, 4))
        posq[16:32, 0] = 10
        posq[48:64, 0] = 10
        posq[80:96, 0] = 10
        data = dict(posq=posq, tiles=np.array([0, 0]), atoms=np.arange(32, 96).reshape(2, 32),
                    exclusions=np.array([[0, 0]]), singles=np.empty((0, 2)), box=np.eye(3)*100,
                    metadata=json.dumps(dict(test="test", precision="single", step=0, num_atoms=96,
                                             periodic=False, cutoff=1., padded_cutoff=1.08)))
        result = analyze(data, chunk=1)
        self.assertEqual(result["padded"]["q"], 0.5)
        self.assertEqual(result["padded"]["q_without_cross_tile_compaction"], 1.)
        self.assertEqual(result["unchanged_exclusion_tiles"], 1)

    def test_skin_and_padding_count_real_packet_work(self):
        posq = np.zeros((64, 4))
        posq[32:, 0] = 5
        posq[32, 0] = 1.04
        data = dict(posq=posq, tiles=np.array([0]), atoms=np.arange(32, 64).reshape(1, 32),
                    exclusions=np.empty((0, 2)), singles=np.empty((0, 2)), box=np.eye(3)*100,
                    metadata=json.dumps(dict(test="test", precision="mixed", step=0, num_atoms=64,
                                             periodic=False, cutoff=1., padded_cutoff=1.08)))
        result = analyze(data)
        self.assertEqual(result["cutoff"]["q"], 0.)
        self.assertEqual(result["padded"]["q"], 1.)
        self.assertEqual(result["padded"]["q_without_packet_padding"], 1/32)
        self.assertEqual(result["verdict"], "KILL")


if __name__ == "__main__":
    unittest.main()
