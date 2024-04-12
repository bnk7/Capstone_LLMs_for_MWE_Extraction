import pandas as pd
import ast
from transformers import AutoTokenizer
import os
from sklearn.model_selection import train_test_split
from collections import Counter


class Data:
    def __init__(self, mwe_type: str = 'all', merge_idiom_other=True):
        """
        :param mwe_type: all (all MWE types), MWE (merge all MWE types),
        V-P_CONSTRUCTION, LIGHT_V, NN_COMP, IDIOM, or OTHER
        :param merge_idiom_other: whether to merge the idiom and other categories
        """
        if mwe_type == 'all':
            labels = ['O', 'B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'B-LIGHT_V', 'I-LIGHT_V',
                      'B-NN_COMP', 'I-NN_COMP', 'B-IDIOM', 'I-IDIOM']
            if not merge_idiom_other:
                labels.extend(['B-OTHER', 'I-OTHER'])
        else:
            labels = ['O', 'B-' + mwe_type, 'I-' + mwe_type]

        self.mwe_type = mwe_type
        self.merge_idiom_other = merge_idiom_other
        self.label_itos = {i: label for (i, label) in enumerate(labels)}
        self.label_stoi = {label: i for (i, label) in enumerate(labels)}
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dataset = self.build_dataset()

    def tokenize_and_format_labels(self, instance: pd.Series) -> pd.Series:
        """
        Format the labels to align with the BERT tokenizer

        :param instance: a data point
        :return: Formatted labels and tokens
        """
        labels = ['O', 'B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'B-LIGHT_V', 'I-LIGHT_V',
                  'B-NN_COMP', 'I-NN_COMP', 'B-IDIOM', 'I-IDIOM', 'B-OTHER', 'I-OTHER']
        label_dict = {label: i for (i, label) in enumerate(labels)}

        old_tags = instance['labels']
        tokenized_input = self.tokenizer(instance['tokens'], is_split_into_words=True)
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
        word_ids = tokenized_input.word_ids()

        prev_word = -1
        new_tags = []
        for word in word_ids:
            if word is None:
                tag = -100
            else:
                tag = old_tags[word]

                # change other to idiom
                if self.merge_idiom_other and tag > 8:
                    tag -= 2
                # change all MWEs to B or I
                if self.mwe_type == 'MWE' and tag > 2:
                    tag = 1 if tag % 2 == 1 else 2
                # change non-target MWE types to O
                elif self.mwe_type not in ['MWE', 'all']:
                    if tag == label_dict['B-' + self.mwe_type]:
                        tag = 1
                    elif tag == ['I-' + self.mwe_type]:
                        tag = 2
                    else:
                        tag = 0

                if word == prev_word and tag % 2 == 1:
                    # change B to I
                    tag += 1
            new_tags.append(tag)
            prev_word = word

        tokenized_input['labels'] = new_tags
        tokenized_input['tokens'] = tokens
        tokenized_input['doc_id'] = instance['doc_id']
        return pd.Series(tokenized_input)

    def build_dataset(self) -> dict[str, pd.DataFrame]:
        """
        Read, process, and split data

        :return: Tokenized train/test/dev split
        """
        if not os.path.exists('data/indexed_bio_labels.csv'):
            write_csv(self.label_stoi)

        dataset = pd.read_csv('data/indexed_bio_labels.csv')
        dataset['tokens'] = dataset['tokens'].map(ast.literal_eval)
        dataset['labels'] = dataset['tag_indices'].map(ast.literal_eval)
        dataset = dataset.drop(['Unnamed: 0', 'tags', 'tag_indices'], axis=1)
        dataset = dataset.apply(lambda row: self.tokenize_and_format_labels(row), axis=1)

        train, test_dev = train_test_split(dataset, test_size=0.2, random_state=24)
        dev, test = train_test_split(test_dev, test_size=0.5, random_state=24)

        dev_counter = Counter()
        for i in dev.index:
            for l in dev.loc[i]['labels']:
                dev_counter[l] += 1
        print(dev_counter)
        print('Baseline dev accuracy:', dev_counter[0]/(sum(dev_counter.values())-dev_counter[-100]))

        return {'train': train.to_dict(orient='list'),
                'test': test.to_dict(orient='list'),
                'dev': dev.to_dict(orient='list')}


def write_csv(label_dict: dict[str, int]) -> None:
    """
    Create a CSV with the tags converted to indices

    :param label_dict: string to index label mapping
    :return: None
    """
    df = pd.read_csv('data/bio_labels.csv')
    df['tokens'] = df['tokens'].map(ast.literal_eval)
    df['tags'] = df['tags'].map(ast.literal_eval)
    df['tag_indices'] = df['tags'].map(lambda x: [label_dict[label] for label in x])
    df.to_csv('data/indexed_bio_labels.csv')


if __name__ == '__main__':
    ds = Data()
    print(ds.dataset)
