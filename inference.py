import os
import torch
from torch.utils.data import DataLoader
import evaluate
import pprint

from model import CustomBert, BertCRF
from utils import get_dataloaders, filter_labels, combine_all
from preprocess import Data

seqeval = evaluate.load("seqeval")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict(loader: DataLoader, model: CustomBert | BertCRF, label_itos: dict[int, str], crf: bool) -> list[list[str]]:
    model.to(device)
    with torch.no_grad():
        predictions = []
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            preds = model.decode(x, mask) if crf else torch.argmax(model(x, mask, smax=True), dim=1)
            preds = filter_labels(preds, label_itos, mask)
            predictions.extend(preds)
    return predictions


def inference_from_pretrained(hyperparameters: dict[str], data=None):
    if data:
        data_instance = data
    else:
        if hyperparameters['mwe_type'] == 'none':
            mode = 'MWE'
        else:
            mode = hyperparameters['mwe_type'].upper()
        data_instance = Data(mwe_type=mode)

    dloaders = get_dataloaders(data_instance, batch_size=hyperparameters['batch_size'])
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
        return predict(dloaders['dev'], model, data_instance.label_itos, hyperparameters['crf'])
    except FileNotFoundError:
        print('No model was found fitting the specifications.')


def predict_independently():
    model_specs = {'nn_comp': {'mwe_type': 'NN_COMP', 'crf': True, 'lr': 0.0001, 'batch_size': 4, 'epochs': 5},
                   'v-p_construction': {'mwe_type': 'all', 'crf': True, 'lr': 0.0001, 'batch_size': 16, 'epochs': 10},
                   'light_v': {'mwe_type': 'all', 'crf': True, 'lr': 0.0001, 'batch_size': 16, 'epochs': 10},
                   'idiom': {'mwe_type': 'all', 'crf': True, 'lr': 0.0001, 'batch_size': 8, 'epochs': 10}}

    data_all = Data(mwe_type='all')

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
            predicted[model] = inference_from_pretrained(model_specs[model], data=data_all)
        else:
            predicted[model] = inference_from_pretrained(model_specs[model])
    combined_predicted = combine_all(predicted)

    # get gold labels
    dloaders = get_dataloaders(data_all, batch_size=16)
    true_labels = []
    for x, mask, y in dloaders['dev']:
        true_labels.extend(filter_labels(y, data_all.label_itos, mask))

    # evaluate
    results = seqeval.compute(predictions=combined_predicted, references=true_labels)
    pprint.pprint(results)


if __name__ == '__main__':
    predict_independently()
