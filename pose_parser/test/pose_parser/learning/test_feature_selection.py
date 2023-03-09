import unittest

import pose_parser.learning.feature_selection as fs


class TestFeatureSelection(unittest.TestCase):
    def test_select_features(self):
        fs.select_features()
