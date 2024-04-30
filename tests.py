import unittest
import pandas as pd
from transformers import AutoTokenizer
from utils import combine
from preprocess import split_tokens
from process_new_data import get_mwe_char_spans


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


class TestSplit(unittest.TestCase):
    def test_all(self):
        df = pd.DataFrame({'test_case': ['beginning_mwe', 'right_mwe', 'left_mwe', 'end_mwe'],
                           'tokens': [['\nOnce', 'upon', 'a', 'time'], ['.\nOnce', 'upon', 'a', 'time'],
                                      ['Once', 'upon', 'a', 'time\n,'], ['Once', 'upon', 'a', 'time\n']],
                           'tags': [['B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'I-IDIOM'],
                                    ['B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'I-IDIOM'],
                                    ['B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'I-IDIOM'],
                                    ['B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'I-IDIOM']]})
        target_df = pd.DataFrame({'test_case': ['beginning_mwe', 'right_mwe', 'left_mwe', 'end_mwe'],
                                  'tokens': [['Once', 'upon', 'a', 'time'], ['.', 'Once', 'upon', 'a', 'time'],
                                             ['Once', 'upon', 'a', 'time', ','], ['Once', 'upon', 'a', 'time']],
                                  'tags': [['B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'I-IDIOM'],
                                           ['O', 'B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'I-IDIOM'],
                                           ['B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'I-IDIOM', 'O'],
                                           ['B-IDIOM', 'I-IDIOM', 'I-IDIOM', 'I-IDIOM']]})
        self.assertTrue(df.apply(split_tokens, axis=1).equals(target_df))


class TestSpans(unittest.TestCase):
    def test_simple(self):
        sentence = 'The uninhabitable house had the power of invisibility.'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenized_input = tokenizer(sentence)

        preds = ['O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP',
                 'O', 'B-LIGHT_V', 'I-LIGHT_V', 'I-LIGHT_V', 'O', 'O', 'O', 'O']
        self.assertEqual(get_mwe_char_spans(preds, tokenized_input),
                         [('NN_COMP', 4, 24), ('LIGHT_V', 28, 41)])

    def test_adjacent(self):
        sentence = 'The uninhabitable house had the power of invisibility.'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenized_input = tokenizer(sentence)

        preds = ['O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP',
                 'B-LIGHT_V', 'I-LIGHT_V', 'I-LIGHT_V', 'I-LIGHT_V', 'O', 'O', 'O', 'O']
        self.assertEqual(get_mwe_char_spans(preds, tokenized_input),
                         [('NN_COMP', 4, 24), ('LIGHT_V', 24, 41)])

    def test_disagreement(self):
        sentence = 'The uninhabitable house had the power of invisibility.'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenized_input = tokenizer(sentence)

        preds = ['O', 'B-NN_COMP', 'O', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP',
                 'B-LIGHT_V', 'I-LIGHT_V', 'I-LIGHT_V', 'I-LIGHT_V', 'O', 'O', 'O', 'O']
        self.assertEqual(get_mwe_char_spans(preds, tokenized_input),
                         [('NN_COMP', 4, 24), ('LIGHT_V', 24, 41)])

    def test_only_i(self):
        sentence = 'The uninhabitable house had the power of invisibility.'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenized_input = tokenizer(sentence)

        preds = ['O', 'B-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP', 'I-NN_COMP',
                 'I-LIGHT_V', 'I-LIGHT_V', 'I-LIGHT_V', 'I-LIGHT_V', 'O', 'O', 'O', 'O']
        self.assertEqual(get_mwe_char_spans(preds, tokenized_input),
                         [('NN_COMP', 4, 24), ('LIGHT_V', 24, 41)])


if __name__ == '__main__':
    unittest.main()
