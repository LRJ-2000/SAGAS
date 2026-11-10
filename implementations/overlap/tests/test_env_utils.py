import unittest
from unittest import mock


class _Wrapper:
    def __init__(self, env):
        self.env = env


with mock.patch.dict(
    "sys.modules",
    {
        "gymnasium": mock.Mock(Wrapper=_Wrapper),
        "numpy": mock.Mock(),
        "ogbenchTL": mock.Mock(),
        "O_utils.datasets": mock.Mock(Dataset=mock.Mock()),
    },
):
    from O_utils.env_utils import ltl_env_render_dimensions, maze_map_type_from_env_name


class EnvUtilsTests(unittest.TestCase):
    def test_maze_map_type_from_state_env_name(self):
        self.assertEqual(maze_map_type_from_env_name("antmaze-giant-stitch-v0"), "giant")
        self.assertEqual(
            maze_map_type_from_env_name("humanoidmaze-medium-stitch-v0"),
            "medium",
        )

    def test_maze_map_type_from_visual_env_name(self):
        self.assertEqual(
            maze_map_type_from_env_name("visual-antmaze-giant-stitch-v0"),
            "giant",
        )
        self.assertEqual(
            maze_map_type_from_env_name("visual-humanoidmaze-large-stitch-v0"),
            "large",
        )

    def test_malformed_env_name_reports_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Cannot infer maze map type"):
            maze_map_type_from_env_name("visual-antmaze")

    def test_ltl_env_render_dimensions_preserve_visual_square_observations(self):
        self.assertEqual(
            ltl_env_render_dimensions("visual-antmaze-giant-stitch-v0", (12, 16), 64),
            (64, 64),
        )

    def test_ltl_env_render_dimensions_keep_state_env_map_aspect_ratio(self):
        self.assertEqual(
            ltl_env_render_dimensions("antmaze-giant-stitch-v0", (12, 16), 480),
            (640, 480),
        )


if __name__ == "__main__":
    unittest.main()
