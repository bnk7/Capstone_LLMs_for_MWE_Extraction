import unittest
from utils import combine


class TestCombine1(unittest.TestCase):

    def test_adjacent(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'B-IDIOM', 'I-IDIOM', 'O', 'O', 'O', 'O', 'O', 'O']),
                         ['O', 'B-IDIOM', 'I-IDIOM', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'])

    def test_front_overlap(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'B-IDIOM', 'I-IDIOM', 'O', 'O', 'O', 'O', 'O']),
                         ['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'])

    def test_misformed_front_overlap(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'I-IDIOM', 'I-IDIOM', 'O', 'O', 'O', 'O', 'O']),
                         ['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'])

    def test_inside(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'B-IDIOM', 'I-IDIOM', 'O', 'O', 'O']),
                         ['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'])

    def test_back_overlap(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'B-IDIOM', 'I-IDIOM', 'O', 'O']),
                         ['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'])

    def test_one_adjacent_one_inside(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'B-IDIOM', 'B-IDIOM', 'I-IDIOM', 'O']),
                         ['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'B-IDIOM', 'I-IDIOM', 'O'])

    def test_misformed_adjacent(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'I-IDIOM', 'O', 'O']),
                         ['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-IDIOM', 'O', 'O'])


class TestCombineAll(unittest.TestCase):
    def test_all_adjacent(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['O', 'B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'B-LIGHT_V', 'I-LIGHT_V', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']),
                         ['O', 'B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP',
                          'B-LIGHT_V', 'I-LIGHT_V', 'O'])

    def test_all_overlap(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'B-LIGHT_V', 'I-LIGHT_V', 'I-LIGHT_V', 'O'],
                                 ['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O', 'O', 'O', 'O', 'O']),
                         ['B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP',
                          'O', 'O', 'O'])

    def test_one_overlap(self):
        self.assertEqual(combine(['O', 'O', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'O', 'O', 'O'],
                                 ['B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                                 ['O', 'O', 'O', 'O', 'O', 'B-IDIOM', 'I-IDIOM', 'B-IDIOM', 'I-IDIOM']),
                         ['B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP',
                          'O', 'B-IDIOM', 'I-IDIOM'])


if __name__ == '__main__':
    unittest.main()
