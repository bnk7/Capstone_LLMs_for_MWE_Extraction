import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import evaluate
import pprint
from datetime import datetime
import argparse

from preprocess import Data
from model import CustomBert, BertCRF

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--eval_every', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--crf', action='store_true')
parser.add_argument('--predict_classes', action='store_true')
args = parser.parse_args()

seqeval = evaluate.load("seqeval")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)


def train(dataloaders: dict[str, DataLoader]) -> None:
    print(f'Training began at {datetime.now()}.')
    classes = 9 if args.predict_classes else 3
    model = BertCRF(num_classes=classes) if args.crf else CustomBert(num_classes=classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        epoch_loss = 0
        for x, mask, y in dataloaders['train']:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            batch_loss = model(x, y, mask) if args.crf else loss_fn(model(x, mask), y)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss.item()
        print(f'Epoch {epoch+1} finished at {datetime.now()}.')
        print('Epoch loss:', epoch_loss)
        if (epoch+1) % args.eval_every == 0:
            evaluate(dataloaders, model)


def evaluate(dataloaders: dict[str, DataLoader], model: CustomBert | BertCRF) -> None:
    model.to(device)
    with torch.no_grad():
        predictions = []
        true_labels = []
        for x, mask, y in dataloaders['dev']:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            preds = model.decode(x, mask) if args.crf else torch.argmax(model(x, mask, smax=True), dim=1)

            # filter out the padding, [CLS], and [SEP]
            for i, labels in enumerate(y):
                masked_labels = []
                masked_preds = []
                for j, label in enumerate(labels):
                    if label.item() != -100:
                        masked_labels.append(data.label_itos[label.item()])
                        predicted = preds[i][j]
                        if isinstance(predicted, int):
                            masked_preds.append(data.label_itos[predicted])
                        else:
                            masked_preds.append(data.label_itos[predicted.item()])

                true_labels.append(masked_labels)
                predictions.append(masked_preds)

        results = seqeval.compute(predictions=predictions, references=true_labels)
        pprint.pprint(results)


if __name__ == '__main__':
    mode = 'merge_idiom_other' if args.predict_classes else 'merge_mwes'
    data = Data(mode=mode)

    # create dictionary of dataloaders
    dloaders = {}
    for split in data.dataset:
        attrs = []
        for attr in ['input_ids', 'attention_mask', 'labels']:
            pad = -100 if attr == 'labels' else 0
            tensor_list = [torch.tensor(lst) for lst in data.dataset[split][attr]]
            padded_2d_tensor = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=pad)
            attrs.append(padded_2d_tensor)
        dataset = torch.utils.data.TensorDataset(attrs[0], attrs[1], attrs[2])
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
        dloaders[split] = dataloader

    train(dloaders)
