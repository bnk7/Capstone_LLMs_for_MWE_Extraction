import torch.nn as nn
from torch.utils.data import DataLoader
import torch


def get_dataloaders(data, batch_size: int) -> dict[str, DataLoader]:
    """
    Create a dictionary of dataloaders

    :param data: instance of Data class
    :param batch_size: batch size
    :return: train, dev, and test dataloaders
    """
    dataloaders = {}
    for split in data.dataset:
        attrs = []
        for attr in ['input_ids', 'attention_mask', 'labels']:
            pad = -100 if attr == 'labels' else 0
            tensor_list = [torch.tensor(lst) for lst in data.dataset[split][attr]]
            padded_2d_tensor = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=pad)
            attrs.append(padded_2d_tensor)
        dataset = torch.utils.data.TensorDataset(attrs[0], attrs[1], attrs[2])
        shuffle = True if split == 'train' else False
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        dataloaders[split] = dataloader
    return dataloaders


def filter_labels(all_labels: torch.Tensor | list[list[int]], label_itos: dict[int, str], msk: torch.tensor) \
        -> list[list[str]]:
    """
    Convert label indices to strings, filtering out padding and special tokens

    :param all_labels: true or predicted labels
    :param label_itos: dictionary mapping label indices to strings
    :param msk: mask tensor
    :return: filtered labels
    """
    filtered_labels = []
    for i, labels in enumerate(all_labels):
        masked = []
        for j, label in enumerate(labels[1:-1]):  # ignore [CLS] at the beginning by starting at the second position
            int_label = label if isinstance(label, int) else label.item()
            # [SEP] is at the last position where the mask is 1,
            # so to exclude it, only add labels for which the following position is not masked
            if msk[i][j+1].item() == 1 and msk[i][j+2].item() == 1:
                masked.append(label_itos[int_label])
        filtered_labels.append(masked)
    return filtered_labels


def add_previous_mwe(beginning_idx: int, curr_idx: int, mwe_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    If we've reached the end of a MWE, add it to the list

    :param beginning_idx: starting index of the potential MWE
    :param curr_idx: ending index of potential MWE + 1
    :param mwe_list: current list of MWE spans
    :return: modified list
    """
    if beginning_idx != -1:
        mwe_list.append((beginning_idx, curr_idx - 1))
    return mwe_list


def get_mwe_token_spans(labels: list[str], mwe_type: str) -> list[tuple[int, int]]:
    """
    Extract the token spans of every MWE of the specified type from the predictions

    :param labels: predictions
    :param mwe_type: type of multiword expression
    :return: token spans of labels of specified type of multiword expression
    """
    labels = [label if mwe_type in label else 'O' for label in labels]
    mwes = []
    # keep track of the current MWE's start index; -1 indicates there is no current MWE
    beginning_idx_curr_mwe = -1
    for i, label in enumerate(labels):
        if label == 'O':
            mwes = add_previous_mwe(beginning_idx_curr_mwe, i, mwes)
            beginning_idx_curr_mwe = -1
        elif label[0] == 'B':
            mwes = add_previous_mwe(beginning_idx_curr_mwe, i, mwes)
            beginning_idx_curr_mwe = i
        elif beginning_idx_curr_mwe == -1:
            beginning_idx_curr_mwe = i
    mwes = add_previous_mwe(beginning_idx_curr_mwe, len(labels), mwes)
    return mwes


def combine(nn_comp: list[str], v_p_construction: list[str], light_v_construction: list[str], idiom: list[str]) \
        -> list[str]:
    """
    Combine separate model predictions for one paragraph into one

    :param nn_comp: noun-noun compound predictions
    :param v_p_construction: verb phrase construction predictions
    :param light_v_construction: light verb construction predictions
    :param idiom: idiom predictions
    :return: combined predictions for one paragraph
    """
    # start with NN compound predictions because they are the most accurate
    combined_labels = nn_comp
    all_mwes = get_mwe_token_spans(combined_labels, 'NN_COMP')
    prediction_dict = {'V-P_CONSTRUCTION': v_p_construction, 'LIGHT_V': light_v_construction, 'IDIOM': idiom}

    for mwe_type in prediction_dict:
        new_labels = prediction_dict[mwe_type]
        new_mwes = get_mwe_token_spans(new_labels, mwe_type)

        # add non-overlapping new mwes
        for span in new_mwes:
            # check if overlap with any span in all_mwes
            overlap = False
            for all_mwe_span in all_mwes:
                if span[0] <= all_mwe_span[0] <= span[1]:  # the end of the new span overlaps
                    overlap = True
                elif span[0] <= all_mwe_span[1] <= span[1]:  # the beginning of the new span overlaps
                    overlap = True
            # if not, add to combined_labels and all_mwes
            if not overlap:
                for i in range(span[0], span[1] + 1):
                    combined_labels[i] = new_labels[i]
                all_mwes.append(span)

    return combined_labels


def combine_all(predictions: dict[str, list[list[str]]]) -> list[list[str]]:
    """
    Combine separate model predictions into one

    :param predictions: model predictions for each MWE type
    :return: combined predictions
    """
    all_labels = []
    for idx, pred in enumerate(predictions['nn_comp']):
        all_labels.append(combine(pred, predictions['v-p_construction'][idx], predictions['light_v'][idx],
                                  predictions['idiom'][idx]))
    return all_labels


def get_label_dicts(mwe_type: str, merge_idiom_other: bool = True) -> tuple[dict[int, str], dict[str, int]]:
    """
    Get mapping between label strings and indices

    :param mwe_type: type of MWE
    :param merge_idiom_other: whether to merge the idiom and other categories
    :return: mappings from label index to string and string to index
    """
    if mwe_type == 'all':
        labels = ['O', 'B-V-P_CONSTRUCTION', 'I-V-P_CONSTRUCTION', 'B-LIGHT_V', 'I-LIGHT_V',
                  'B-NN_COMP', 'I-NN_COMP', 'B-IDIOM', 'I-IDIOM']
        if not merge_idiom_other:
            labels.extend(['B-OTHER', 'I-OTHER'])
    else:
        labels = ['O', 'B-' + mwe_type, 'I-' + mwe_type]
    return {i: label for (i, label) in enumerate(labels)}, {label: i for (i, label) in enumerate(labels)}
