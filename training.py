import random
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from data import LeafDataset
from models import Classifier, parameter_counts


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metrics(model, loader, device, classes):
    model.eval()
    truth, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(1).cpu().tolist()
            predictions.extend(predicted)
            truth.extend(labels.tolist())
    if not truth:
        raise ValueError('Evaluation split is empty.')
    return {'accuracy': accuracy_score(truth, predictions),
            'macro_f1': f1_score(truth, predictions, labels=list(range(classes)),
                                 average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(truth, predictions,
                                                labels=list(range(classes))).tolist()}


def fit(manifest, config, *, epochs, seed, size, device, architecture, pretrained=True):
    seed_everything(seed)
    train = LeafDataset(manifest, 'train', size)
    val = LeafDataset(manifest, 'val', size)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=config.batch_size, shuffle=True,
                        num_workers=0, generator=generator)
    validation = DataLoader(val, batch_size=config.batch_size, num_workers=0)
    model = Classifier(len(manifest['classes']), config, architecture, pretrained).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=config.learning_rate, momentum=0.9,
                                   weight_decay=config.weight_decay)
    else:
        raise ValueError('Unknown optimizer.')
    loss_fn = nn.CrossEntropyLoss()
    history, best_weights, best_f1 = [], None, -1
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        score = metrics(model, validation, device, len(manifest['classes']))
        history.append({'epoch': epoch + 1, 'train_loss': total_loss / len(train),
                        'val_accuracy': score['accuracy'], 'val_macro_f1': score['macro_f1']})
        if score['macro_f1'] > best_f1:
            best_f1 = score['macro_f1']
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_weights is None:
        raise ValueError('epochs must be positive.')
    model.load_state_dict(best_weights)
    return model, {'best_val_macro_f1': best_f1, 'history': history,
                   'train_seconds': time.perf_counter() - started,
                   'parameters': parameter_counts(model)}
