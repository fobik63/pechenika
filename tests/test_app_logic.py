import unittest

from app import Detection, choose_target, click_point, toggle_pause


class ClickerTests(unittest.TestCase):
    def test_space_toggles_on_key_down_edge(self):
        paused, down = toggle_pause(False, False, True)
        self.assertTrue(paused)
        paused, down = toggle_pause(paused, down, True)
        self.assertTrue(paused)
        paused, down = toggle_pause(paused, down, False)
        paused, down = toggle_pause(paused, down, True)
        self.assertFalse(paused)

    def test_priority_and_upgrade_cooldown(self):
        game = Detection("game", 0.9, (0, 0, 20, 20))
        upgrade = Detection("upgrade", 0.8, (20, 20, 40, 40))
        ad = Detection("ads_close", 0.7, (100, 10, 120, 30))
        self.assertEqual(choose_target([game, upgrade, ad], 10, 11, 4), ad)
        self.assertEqual(choose_target([game, upgrade], 10, 11, 4), game)
        self.assertEqual(choose_target([game, upgrade], 10, 14, 4), upgrade)

    def test_ad_click_point(self):
        self.assertEqual(click_point(Detection("ads_close", 0.9, (100, 20, 200, 120))), (190, 30))
        self.assertEqual(click_point(Detection("ads_close", 0.9, (100, 20, 120, 40))), (110, 30))


if __name__ == "__main__":
    unittest.main()
