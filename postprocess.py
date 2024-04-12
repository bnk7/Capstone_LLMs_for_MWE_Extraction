import re
import pandas as pd
import evaluate
import pprint
import ast


def get_pred_and_gold_labels(df_row: pd.Series,
                             gpt_output: str,
                             label_set: list[str],
                             mwe_type_dict: dict[str, str]) -> tuple[list[str], list[str]]:
    """
    Convert GPT response and gold labels to BIO tags

    :param df_row: relevant row of the dataframe
    :param gpt_output: ordered list response from GPT
    :param label_set: all tags
    :param mwe_type_dict: map from MWE types to abbreviations
    :return: list of predicted labels and list of true labels
    """
    paragraph = df_row['original_text'].iloc[0]
    true = df_row['all'].iloc[0]
    true = [word.replace('OTHER', 'IDIOM') for word in true]  # merge these tags

    # extract MWEs from GPT output
    lines = gpt_output.split('\n')
    mwes = []
    for line in lines:
        parts = re.split(r'\s*\|\s*', line)
        # only process if the format is correct
        if len(parts) == 3:
            is_mwe = True if parts[1].lower() == 'true' else False
            if is_mwe:
                # remove the list label, e.g. '1. '
                expression_match = re.fullmatch(r'\d+[.)](\s+|\\t)(.+)', parts[0])
                if expression_match:
                    expression = expression_match.group(2)
                else:
                    is_mwe = False
            if is_mwe:
                # find the last string in parentheses (without parentheses in it itself)
                mwe_type_match = re.findall(r'(?<=\()[^()]*(?=\))', parts[2])
                # check that it is a valid MWE type
                if len(mwe_type_match) > 0 and mwe_type_match[-1].lower() in mwe_type_dict:
                    mwe_type = mwe_type_dict[mwe_type_match[-1].lower()]
                # if no valid type is specified, don't count it as a MWE
                else:
                    is_mwe = False
            if is_mwe:
                mwes.append((expression, mwe_type))

    # label in BIO format
    labeled_paragraph = paragraph
    for expression, mwe_type in mwes:
        num_tokens = len(expression.split())
        label = 'B-' + mwe_type + (' I-' + mwe_type) * (num_tokens - 1)
        labeled_paragraph = re.sub(expression, label, labeled_paragraph)
    predicted = [label if label in label_set else 'O' for label in labeled_paragraph.split(' ')]
    return predicted, true


if __name__ == '__main__':
    mwe_types = {
        'noun-noun compound': 'NN_COMP', 'noun noun compound': 'NN_COMP',
        'light verb construction': 'LIGHT_V',
        'verb-particle construction': 'V-P_CONSTRUCTION', 'verb particle construction': 'V-P_CONSTRUCTION',
        'idiom': 'IDIOM'
    }
    prefixes = ['B-', 'I-']
    suffixes = set(mwe_types.values())
    labels = [prefix + suffix for prefix in prefixes for suffix in suffixes]

    df = pd.read_csv('data/annotations.csv', usecols=['doc_id', 'original_text', 'all'])
    df['all'] = df['all'].map(ast.literal_eval)

    with open('gpt_responses.txt') as f:
        responses = f.read()
    responses = responses.split('ID: ')
    # isolate ID
    responses = {response.split('\n', maxsplit=1)[0]: response.split('\n', maxsplit=1)[1] for response in responses
                 if len(response.split('\n', maxsplit=1)) > 1}

    # collect predictions and gold labels
    all_preds = []
    all_gold = []
    for doc_id in responses:
        row = df[df['doc_id'] == int(doc_id)]
        preds, gold = get_pred_and_gold_labels(row, responses[doc_id], labels, mwe_types)
        all_preds.append(preds)
        all_gold.append(gold)

    # evaluate
    seqeval = evaluate.load("seqeval")
    evaluation = seqeval.compute(predictions=all_preds, references=all_gold)
    pprint.pprint(evaluation)
