import math
import unittest

from S_utils.low_level_tracking import (
    LowLevelTrackingConfig,
    active_forbidden_labels,
    compute_tracking_command,
    direction_to_skill,
    semantic_repulsion_delta,
    select_tracking_target,
    vector_norm,
)


class LowLevelTrackingTests(unittest.TestCase):
    def test_direction_to_skill_normalizes_and_smooths(self):
        skill, next_prev = direction_to_skill([3.0, 4.0])
        self.assertAlmostEqual(vector_norm(skill), 1.0)
        self.assertEqual(next_prev, skill)

        smoothed, _ = direction_to_skill(
            [0.0, 1.0],
            prev_skill=[1.0, 0.0],
            smoothing_beta=0.5,
        )
        self.assertAlmostEqual(smoothed[0], 1.0 / math.sqrt(2.0), places=6)
        self.assertAlmostEqual(smoothed[1], 1.0 / math.sqrt(2.0), places=6)

    def test_active_forbidden_labels_uses_current_edge_window(self):
        edge_labels = [{"a"}, {"b", "c"}, {"d"}]
        self.assertEqual(active_forbidden_labels(edge_labels, 0, 2), {"a", "b", "c"})
        self.assertEqual(active_forbidden_labels(edge_labels, 1, 3), {"b", "c", "d"})
        self.assertEqual(active_forbidden_labels(edge_labels, 2, 2), set())
        self.assertEqual(active_forbidden_labels(edge_labels, 0, 1, use_all=True), {"a", "b", "c", "d"})

    def test_semantic_repulsion_points_away_from_nearest_forbidden_anchor(self):
        repulsion = semantic_repulsion_delta(
            phi_obs=[0.0, 0.0],
            forbidden_labels={"bad"},
            anchor_positions_by_label={"bad": [[0.2, 0.0], [3.0, 0.0]]},
            radius=1.0,
            gain=0.5,
            power=1.0,
            max_norm=0.5,
        )
        self.assertLess(repulsion[0], 0.0)
        self.assertAlmostEqual(repulsion[1], 0.0)
        self.assertLessEqual(vector_norm(repulsion), 0.5 + 1e-8)

    def test_compute_tracking_command_keeps_waypoint_direction_primary(self):
        command = compute_tracking_command(
            phi_obs=[0.0, 0.0],
            target_latent=[0.0, 1.0],
            active_negatives={"bad"},
            anchor_positions_by_label={"bad": [[0.2, 0.0]]},
            config=LowLevelTrackingConfig(
                semantic_repulsion_radius=1.0,
                semantic_repulsion_gain=0.5,
                semantic_repulsion_max_norm=0.25,
            ),
        )
        self.assertLess(command.repulsion_delta[0], 0.0)
        self.assertLess(command.command_delta[0], command.base_delta[0])
        self.assertGreater(command.command_delta[1], 0.0)
        self.assertAlmostEqual(vector_norm(command.skill), 1.0)
        self.assertTrue(command.repulsion_active)
        self.assertEqual(command.active_forbidden_labels, {"bad"})

    def test_select_tracking_target_skips_only_before_pending_checkpoint(self):
        selection = select_tracking_target(
            current_latent=[0.0, 0.0],
            waypoint_latents=[
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 0.0],
            ],
            current_index=0,
            local_reachability_radius=1.0,
            semantic_checkpoint_indices={2},
        )
        self.assertEqual(selection.target_index, 2)
        self.assertEqual(selection.skipped_indices, [0, 1])
        self.assertEqual(selection.blocked_by_checkpoint, 2)

        verified_selection = select_tracking_target(
            current_latent=[0.0, 0.0],
            waypoint_latents=[
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 0.0],
            ],
            current_index=0,
            local_reachability_radius=1.0,
            semantic_checkpoint_indices={2},
            verified_checkpoint_indices={2},
        )
        self.assertEqual(verified_selection.target_index, 3)
        self.assertEqual(verified_selection.skipped_indices, [0, 1, 2])

    def test_select_tracking_target_keeps_current_unverified_checkpoint(self):
        selection = select_tracking_target(
            current_latent=[0.0, 0.0],
            waypoint_latents=[
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
            ],
            current_index=1,
            local_reachability_radius=1.0,
            semantic_checkpoint_indices={1},
        )
        self.assertEqual(selection.target_index, 1)
        self.assertEqual(selection.skipped_indices, [])
        self.assertEqual(selection.blocked_by_checkpoint, 1)


if __name__ == "__main__":
    unittest.main()
