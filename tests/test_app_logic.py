import unittest

from app import Config, Detection, choose_target, toggle_pause


class ControlTests(unittest.TestCase):
    def test_space_toggles_only_on_press_edge(self):
        paused, down, changed = toggle_pause(False, False, True)
        self.assertEqual((paused, down, changed), (True, True, True))
        paused, down, changed = toggle_pause(paused, down, True)
        self.assertEqual((paused, down, changed), (True, True, False))
        paused, down, changed = toggle_pause(paused, down, False)
        self.assertEqual((paused, down, changed), (True, False, False))
        paused, down, changed = toggle_pause(paused, down, True)
        self.assertEqual((paused, down, changed), (False, True, True))

    def test_upgrade_cooldown_does_not_block_cookie(self):
        config = Config()
        upgrade = Detection("upgrade", 0.9, 1, 1)
        cookie = Detection("game", 0.8, 2, 2)
        detections = [cookie, upgrade]
        self.assertEqual(choose_target(detections, {}, 10, config), upgrade)
        self.assertEqual(
            choose_target(detections, {"upgrade": 10}, 11, config), cookie
        )
        self.assertEqual(
            choose_target(detections, {"upgrade": 10}, 18, config), upgrade
        )

    def test_ads_and_bonus_keep_priority(self):
        config = Config()
        detections = [
            Detection("game", 0.99, 1, 1),
            Detection("upgrade", 0.9, 2, 2),
            Detection("bonus", 0.8, 3, 3),
            Detection("ads_close", 0.7, 4, 4),
        ]
        self.assertEqual(choose_target(detections, {}, 10, config).name, "ads_close")
        self.assertIsNone(choose_target(detections, {"ads_close": 10}, 10.1, config))
        self.assertEqual(
            choose_target(detections[0:3], {}, 10, config).name, "bonus"
        )


if __name__ == "__main__":
    unittest.main()
