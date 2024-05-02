import pandas as pd
import ast
from transformers import AutoTokenizer
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from utils import get_label_dicts


class Data:
    def __init__(self, mwe_type: str = 'all', merge_idiom_other: bool = True):
        """
        :param mwe_type: all (all MWE types), MWE (merge all MWE types), V-P_CONSTRUCTION, LIGHT_V, NN_COMP, IDIOM, or OTHER
        :param merge_idiom_other: whether to merge the idiom and other categories
        """
        self.mwe_type = mwe_type
        self.merge_idiom_other = merge_idiom_other
        self.label_itos, self.label_stoi = get_label_dicts(mwe_type, merge_idiom_other)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dataset = self.build_dataset()

    def tokenize_and_format_labels(self, instance: pd.Series, label_dict: dict[str, int]) -> pd.Series:
        """
        Format the labels to align with the BERT tokenizer

        :param instance: a data point
        :param label_dict: mapping from label string to index
        :return: Formatted labels and tokens
        """
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
                    elif tag == label_dict['I-' + self.mwe_type]:
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
        label_dict = get_label_dicts('all', False)[1]
        if not os.path.exists('data/indexed_bio_labels.csv'):
            write_csv(label_dict)

        dataset = pd.read_csv('data/indexed_bio_labels.csv')
        dataset['tokens'] = dataset['tokens'].map(ast.literal_eval)
        dataset['labels'] = dataset['tag_indices'].map(ast.literal_eval)
        dataset = dataset.drop(['Unnamed: 0', 'tags', 'tag_indices'], axis=1)
        dataset = dataset.apply(lambda row: self.tokenize_and_format_labels(row, label_dict), axis=1)

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

    :param label_dict: mapping from label string to index
    :return: None
    """
    df = pd.read_csv('data/bio_labels.csv')
    df['tokens'] = df['tokens'].map(ast.literal_eval)
    df['tags'] = df['tags'].map(ast.literal_eval)
    df = df.apply(split_tokens, axis=1)
    df['tag_indices'] = df['tags'].map(lambda x: [label_dict[label] for label in x])
    df.to_csv('data/indexed_bio_labels.csv')


def split_tokens(row: pd.Series) -> pd.Series:
    """
    Split tokens around whitespace and update the labels to match

    :param row: row of a DataFrame with token and tag lists
    :return: updated row
    """
    new_tokens = []
    new_labels = []
    for i, token in enumerate(row['tokens']):
        split_token = token.split()
        curr_label = row['tags'][i] if 'tags' in row.index else 'O'
        if len(split_token) > 1:
            new_tokens.extend(split_token)
            if curr_label == 'O':
                new_labels.extend([curr_label] * len(split_token))
            elif curr_label.startswith('B'):
                new_labels.extend(['O'] * (len(split_token) - 1))
                new_labels.append(curr_label)
            else:
                new_labels.append(curr_label)
                new_labels.extend(['O'] * (len(split_token) - 1))
        elif len(token) < 1:
            # restore the @ redaction symbol to the dataset (it was turned into spaces for another project)
            new_tokens.append('@')
            new_labels.append(curr_label)
        elif len(split_token) == 1:
            new_tokens.append(split_token[0])
            new_labels.append(curr_label)
    row['tokens'] = new_tokens
    row['tags'] = new_labels
    return row


if __name__ == '__main__':
    ds = Data()
    # print(ds.dataset)
