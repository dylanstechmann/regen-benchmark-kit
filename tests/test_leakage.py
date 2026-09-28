import unittest

from regenbench.leakage import donor_tag_fixture, leakage_gap


class LeakageDiagnosticTests(unittest.TestCase):
    def test_random_rows_beat_donor_holdout_on_the_tag_fixture(self):
        report = leakage_gap(donor_tag_fixture(n_donors=10, replicates=4, seed=0), folds=5, seed=0)
        self.assertTrue(report["not_a_biological_result"])
        self.assertGreater(report["random_row_logistic_accuracy"], 0.9)
        self.assertLess(report["group_holdout_logistic_accuracy"], 0.7)
        self.assertGreater(report["gap_random_minus_group"], 0.2)
