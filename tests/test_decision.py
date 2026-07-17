import unittest

from decision import LABEL_A, LABEL_B, OTHER, UNCERTAIN, decide


class DecisionTest(unittest.TestCase):
    def test_safety_gates_and_labels(self):
        cases = [
            ((0.70, 0.30, 80, 0.95), LABEL_A),
            ((0.31, 0.69, 80, 0.95), LABEL_B),
            ((0.55, 0.53, 80, 0.95), UNCERTAIN),
            ((0.20, 0.18, 80, 0.95), OTHER),
            ((0.70, 0.30, 5, 0.95), UNCERTAIN),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(decide(*args), expected)


if __name__ == "__main__":
    unittest.main()
