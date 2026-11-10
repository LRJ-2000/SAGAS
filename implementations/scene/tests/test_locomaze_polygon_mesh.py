import unittest

import numpy as np

from ogbenchTL.locomaze.maze import (
    _hidden_polygon_mesh_data,
    _polygon_mesh_data,
)


class LocomazePolygonMeshTest(unittest.TestCase):
    def test_hidden_polygon_mesh_stays_near_maze_extent(self):
        vertices, faces = _hidden_polygon_mesh_data(max_vertices=12, max_faces=10)

        self.assertEqual(vertices.shape, (12, 3))
        self.assertEqual(faces.shape, (10, 3))
        self.assertGreaterEqual(float(vertices[:, 2].min()), 0.0)
        self.assertLessEqual(float(vertices[:, 2].max()), 0.0)
        self.assertTrue(np.all(faces == np.array([0, 1, 2], dtype=np.int32)))

    def test_polygon_mesh_padding_uses_valid_faces(self):
        vertices, faces, center_x, center_y = _polygon_mesh_data(
            [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
            max_vertices=12,
            max_faces=10,
        )

        self.assertEqual(vertices.shape, (12, 3))
        self.assertEqual(faces.shape, (10, 3))
        self.assertAlmostEqual(center_x, 1.0)
        self.assertAlmostEqual(center_y, 0.5)
        self.assertTrue(np.all(faces >= 0))
        self.assertTrue(np.all(faces < 12))
        for face in faces:
            self.assertEqual(len(set(int(i) for i in face)), 3)


if __name__ == "__main__":
    unittest.main()
