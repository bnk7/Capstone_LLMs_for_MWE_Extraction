import torch
from torch.utils.data import DataLoader
import evaluate
import pprint
from datetime import datetime
import argparse
import os

from preprocess import Data
from model import CustomBert, BertCRF
from utils import get_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--eval_every', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--crf', action='store_true')
parser.add_argument('--mwe_type', choices=['all', 'none', 'v-p_construction', 'light_v', 'nn_comp', 'idiom', 'other'],
                    default='all')
args = parser.parse_args()

seqeval = evaluate.load("seqeval")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)


def train(dataloaders: dict[str, DataLoader], data) -> None:
    print(f'Training began at {datetime.now()}.')
    classes = 9 if args.mwe_type == 'all' else 3
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
            evaluate(dataloaders, model, data)

    # save model
    filename = 'bert_crf_' if args.crf else 'bert_'
    filename += '_'.join([args.mwe_type, str(args.lr), str(args.batch_size), str(args.epochs)])
    torch.save(model.state_dict(), os.path.join('saved_models', filename + '.pt'))


def evaluate(dataloaders: dict[str, DataLoader], model: CustomBert | BertCRF, data) -> None:
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
    if args.mwe_type == 'all':
        mode = 'all'
    elif args.mwe_type == 'none':
        mode = 'MWE'
    else:
        mode = args.mwe_type.upper()
    data_instance = Data(mwe_type=mode)
    dloaders = get_dataloaders(data_instance, args.batch_size)
    train(dloaders, data_instance)
