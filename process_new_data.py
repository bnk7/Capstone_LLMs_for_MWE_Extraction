from transformers import AutoTokenizer, BatchEncoding
import torch
import os
from model import CustomBert, BertCRF
from utils import filter_labels, combine_all

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_mwe_char_spans(labels: list[str], tokenized: BatchEncoding) -> list[tuple[str, int, int]]:
    """
    Extract the character spans of every MWE from the predictions

    :param labels: predictions
    :param tokenized:
    :return: MWE types and character spans
    """

    # change labels to be per word rather than per token (sub-word)
    # the label of the word is equal to the label of its first token
    word_preds = []
    word_ids = tokenized.word_ids()[1:-1]  # exclude start and end
    for i, word_id in enumerate(word_ids):
        if i == 0 or word_id != word_ids[i - 1]:
            word_preds.append(labels[i])

    # extract the MWE character spans
    mwes = []
    # keep track of the current MWE's start index; -1 indicates there is no current MWE
    beginning_idx_curr_mwe = -1
    curr_mwe_type = ''
    for i, label in enumerate(word_preds):
        char_span = tokenized.word_to_chars(i)
        if label == 'O':
            mwes = add_previous_mwe_with_type(beginning_idx_curr_mwe, char_span[0], curr_mwe_type, mwes)
            beginning_idx_curr_mwe = -1
            curr_mwe_type = ''
        elif label[0] == 'B':
            mwes = add_previous_mwe_with_type(beginning_idx_curr_mwe, char_span[0], curr_mwe_type, mwes)
            beginning_idx_curr_mwe = char_span[0]
            curr_mwe_type = label[2:]
        # account for MWEs missing the B label
        elif beginning_idx_curr_mwe == -1 or curr_mwe_type != label[2:]:
            mwes = add_previous_mwe_with_type(beginning_idx_curr_mwe, char_span[0], curr_mwe_type, mwes)
            beginning_idx_curr_mwe = char_span[0]
            curr_mwe_type = label[2:]
    mwes = add_previous_mwe_with_type(beginning_idx_curr_mwe, len(labels), curr_mwe_type, mwes)
    return mwes


def add_previous_mwe_with_type(beginning_idx: int, curr_idx: int, mwe_type: str, mwe_list) -> list[tuple[str, int, int]]:
    """
    If we've reached the end of a MWE, add it to the list

    :param beginning_idx: starting index of the potential MWE
    :param curr_idx: ending index of potential MWE + 1
    :param mwe_type: type of MWE
    :param mwe_list: current list of MWE spans
    :return: modified list
    """
    if beginning_idx != -1:
        mwe_list.append((mwe_type, beginning_idx, curr_idx - 1))
    return mwe_list


def predict(tokenized: BatchEncoding, model: CustomBert | BertCRF, label_itos: dict[int, str], crf: bool):
    model.to(device)
    with torch.no_grad():
        x = torch.tensor(tokenized['input_ids']).unsqueeze(dim=0)
        x = x.to(device)
        mask = torch.tensor(tokenized['attention_mask']).unsqueeze(dim=0)
        mask = mask.to(device)
        preds = model.decode(x, mask) if crf else torch.argmax(model(x, mask, smax=True), dim=1)
        predictions = filter_labels(preds, label_itos, mask)
    return predictions


def inference_from_pretrained(hyperparameters, tokenized, label_itos):
    classes = 9 if hyperparameters['mwe_type'] == 'all' else 3
    model = BertCRF(num_classes=classes) if hyperparameters['crf'] else CustomBert(num_classes=classes)
    filename = 'bert_crf_' if hyperparameters['crf'] else 'bert_'
    filename += '_'.join([hyperparameters['mwe_type'], str(hyperparameters['lr']), str(hyperparameters['batch_size']),
                          str(hyperparameters['epochs'])])
    try:
        if device == 'cpu':
            model.load_state_dict(torch.load(os.path.join('saved_models', filename + '.pt'),
                                             map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(os.path.join('saved_models', filename + '.pt')))
        return predict(tokenized, model, label_itos, hyperparameters['crf'])
    except FileNotFoundError:
        print('No model was found fitting the specifications.')


def predict_independently(tokenized):
    model_specs = {'nn_comp': {'mwe_type': 'NN_COMP', 'crf': True, 'lr': 0.0001, 'batch_size': 4, 'epochs': 5},
                   'v-p_construction': {'mwe_type': 'all', 'crf': True, 'lr': 0.0001, 'batch_size': 16, 'epochs': 10},
                   'light_v': {'mwe_type': 'all', 'crf': True, 'lr': 0.0001, 'batch_size': 16, 'epochs': 10},
                   'idiom': {'mwe_type': 'all', 'crf': True, 'lr': 0.0001, 'batch_size': 8, 'epochs': 10}}

    # only predict once per model for efficiency
    predicted = {}
    for model in model_specs:
        copied_prediction = None
        for completed_mwe_type in predicted:
            if model_specs[model] == model_specs[completed_mwe_type]:
                copied_prediction = predicted[completed_mwe_type]
        if copied_prediction:
            predicted[model] = copied_prediction
        elif model_specs[model]['mwe_type'] == 'all':
            predicted[model] = inference_from_pretrained(model_specs[model], tokenized, get_label_dict('all'))
        else:
            predicted[model] = inference_from_pretrained(model_specs[model], tokenized,
                                                         get_label_dict(model_specs[model]['mwe_type']))
    combined_predicted = combine_all(predicted)[0]
    return combined_predicted


def get_label_dict(mwe_type):
    if mwe_type == 'all':
        labels = ['O', 'B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'B-LIGHT_V', 'I-LIGHT_V',
                  'B-NN_COMP', 'I-NN_COMP', 'B-IDIOM', 'I-IDIOM']
    else:
        labels = ['O', 'B-' + mwe_type, 'I-' + mwe_type]
    return {i: label for (i, label) in enumerate(labels)}


if __name__ == '__main__':
    user_input = 'This is a test of a multiword expression tokenizer.'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_input = tokenizer(user_input)
    preds = predict_independently(tokenized_input)
    spans = get_mwe_char_spans(preds, tokenized_input)
    print(spans)
