import random
import copy
from time import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class DatasetSplit(Dataset):
    def __init__(self, dataset, indices=None):
        if indices is None:
            indices = range(len(dataset))
        self.dataset = dataset
        self.indices = indices
        for i in indices:
            assert i < len(dataset)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def shuffle_tensors(*tensors):
    assert len(tensors) > 0
    output = []
    count = len(tensors[0])
    indices = torch.randperm(count)
    for tensor in tensors:
        assert len(tensor) == count
        output.append(tensor[indices])
    return output


def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2


def shuffle_dataset(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return DatasetSplit(dataset, indices)


def split_dataset(dataset, ratio=0.7, shuffle=False):
    if type(ratio) is not tuple:
        ratio = (ratio, 1 - ratio)
    ratio = list(ratio)
    assert sum(ratio) <= 1
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    splits = []
    for i in range(len(ratio)):
        start = int(sum(ratio[:i]) * len(dataset))
        end = int(sum(ratio[:i + 1]) * len(dataset))
        split = indices[start:end]
        splits.append(DatasetSplit(dataset, split))
    return tuple(splits)


def _epoch(scope, training=False):
    model = scope["model"]
    optimizers = scope["optimizers"]
    loss_funcs = scope["loss_funcs"]
    loader = scope["loader"]
    scope = copy.copy(scope)
    total_metrics = {}
    total_loss = len(loss_funcs) * [0]
    scope["total_metrics"] = total_metrics
    scope["total_loss"] = total_loss
    if training:
        model.train()
    else:
        model.eval()
    iterator = tqdm(loader)
    iteration = 1
    for batch in iterator:
        metrics = {}
        scope["iteration"] = iteration
        scope["batch"] = batch
        scope["metrics"] = metrics
        if "process_batch" in scope and scope["process_batch"] is not None:
            batch = scope["process_batch"](batch)
        if "device" in scope and scope["device"] is not None:
            batch = [tensor.to(scope["device"]) for tensor in batch]
        losses = len(loss_funcs) * [None]
        outputs = len(loss_funcs) * [None]
        for i in range(len(loss_funcs)):
            scope["model_id"] = i
            scope["model"] = model
            scope["optimizer"] = optimizers[i]
            scope["loss_func"] = loss_funcs[i]
            loss, output = loss_funcs[i](model, batch, scope)
            if training:
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
            total_loss[i] += loss.item()
            losses[i] = loss.item()
            outputs[i] = output
        scope["losses"] = losses
        scope["outputs"] = outputs
        scope["loss"] = losses[0]
        scope["output"] = outputs[0]
        scope["total_loss"] = total_loss
        if "on_batch" in scope and scope["on_batch"] is not None:
            scope["on_batch"](scope)
        for key in metrics:
            if key not in total_metrics:
                total_metrics[key] = metrics[key]
            else:
                total_metrics[key] += metrics[key]
        iteration += 1
    for i in range(len(loss_funcs)):
        total_loss[i] = total_loss[i] / iteration
    return total_loss, total_metrics


def train(model, loss_funcs, train_dataset, val_dataset, optimizers,
          epochs=100, batch_size=256, device=0, print_function=print,
          on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None):
    # Reseting tqdm
    if hasattr(tqdm, "_instances"):
        tqdm._instances.clear()
    # Security Checks
    if type(loss_funcs) is not list and type(loss_funcs) is not tuple:
        loss_funcs = [loss_funcs]
    if type(optimizers) is not list and type(optimizers) is not tuple:
        optimizers = [optimizers]
    assert len(loss_funcs) > 0
    assert len(optimizers) > 0
    if len(optimizers) == 1 and len(loss_funcs) != 1:
        optimizers = len(loss_funcs) * optimizers
    elif len(loss_funcs) == 1 and len(optimizers) != 1:
        loss_funcs = len(optimizers) * loss_funcs
    assert len(loss_funcs) == len(optimizers)
    # Moving models to device
    model = model.to(device)
    # Creating dataset loaders
    train_loader = None if train_dataset is None else torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                                                  shuffle=True)
    val_loader = None if val_dataset is None else torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                                              shuffle=False)
    # Generating Scope
    scope = {
        "model": model,
        "loss_func": loss_funcs[0],
        "loss_funcs": loss_funcs,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "optimizer": optimizers[0],
        "optimizers": optimizers,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }
    # Epochs
    for epoch_id in range(1, epochs + 1):
        scope["epoch"] = epoch_id
        print_function("Epoch #" + str(epoch_id))
        # Training
        if train_loader is not None:
            scope["dataset"] = train_dataset
            scope["loader"] = train_loader
            scope["on_batch"] = on_train_batch
            train_loss, train_metrics = _epoch(scope, True)
            scope["train_losses"] = train_loss
            scope["train_loss"] = train_loss[0]
            scope["train_metrics"] = train_metrics
            if len(loss_funcs) == 1:
                print_function("\tTrain Loss = " + str(train_loss[0]))
            else:
                for i in range(len(loss_funcs)):
                    print_function("\tTrain Loss " + str(i) + " = " + str(train_loss[i]))
            for key in train_metrics:
                print_function("\tTrain " + key + " = " + str(train_metrics[key]))
            if on_train_epoch is not None:
                on_train_epoch(scope)
        # Validation
        if val_loader is not None:
            scope["dataset"] = val_dataset
            scope["loader"] = val_loader
            scope["on_batch"] = on_val_batch
            with torch.no_grad():
                val_loss, val_metrics = _epoch(scope, False)
            scope["val_losses"] = val_loss
            scope["val_loss"] = val_loss[0]
            scope["val_metrics"] = val_metrics
            if len(loss_funcs) == 1:
                print_function("\tValidation Loss = " + str(val_loss[0]))
            else:
                for i in range(len(loss_funcs)):
                    print_function("\tValidation Loss " + str(i) + " = " + str(val_loss[i]))
            for key in val_metrics:
                print_function("\tValidation " + key + " = " + str(val_metrics[key]))
            if on_val_epoch is not None:
                on_val_epoch(scope)
        # Clearing variables
        del scope["dataset"]
        del scope["loader"]
        del scope["on_batch"]
        # Saving model
        if after_epoch is not None:
            after_epoch(scope)
    return model
