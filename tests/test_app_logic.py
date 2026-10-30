import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import Detection, affordable_upgrades, choose_target, click_point, toggle_pause


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
        self.assertEqual(choose_target([game, upgrade, ad], [upgrade], 10, 11, 4), ad)
        self.assertEqual(choose_target([game, upgrade], [], 10, 14, 4), game)
        self.assertEqual(choose_target([game, upgrade], [upgrade], 10, 11, 4), game)
        self.assertEqual(choose_target([game, upgrade], [upgrade], 10, 14, 4), upgrade)

    def test_random_choice_among_affordable_upgrades(self):
        game = Detection("game", 0.9, (0, 0, 20, 20))
        first = Detection("upgrade", 1.0, (100, 100, 100, 100))
        second = Detection("upgrade", 1.0, (100, 170, 100, 170))
        with patch("app.random.choice", return_value=second) as choice:
            self.assertEqual(choose_target([game], [first, second], 0, 10, 4), second)
        choice.assert_called_once_with([first, second])

    def test_green_shop_price_required(self):
        hsv = np.full((600, 800, 3), (25, 30, 180), dtype=np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.putText(image, "202", (300, 322), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 220), 2)
        monitor = {"left": 0, "top": 0}
        self.assertEqual(affordable_upgrades(image, monitor), [])
        cv2.putText(image, "93", (300, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 0), 2)
        self.assertEqual(len(affordable_upgrades(image, monitor)), 1)

    def test_ad_click_point(self):
        self.assertEqual(click_point(Detection("ads_close", 0.9, (100, 20, 200, 120))), (190, 30))
        self.assertEqual(click_point(Detection("ads_close", 0.9, (100, 20, 120, 40))), (110, 30))


if __name__ == "__main__":
    unittest.main()
