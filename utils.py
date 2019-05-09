import torch
import torch.nn as nn

import csv
from tqdm import tqdm
from scipy.misc import imsave


use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

if use_cuda:
    cuda_0 = 'cuda'


def init_xavier_uniform(model):
    for w in model.parameters():
        if len(w.shape) > 1:
            nn.init.xavier_uniform_(w)

    return model


def predict(model, testloader, split=False, idx_s=0, idx_e=50000):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(testloader):
            if split:
                if batch_id == 0:
                    batch_size = len(inputs)
                idx = batch_id * batch_size
                if idx < idx_s or idx >= idx_e: continue
            all_labels.append(labels)
            if use_cuda:
                inputs = inputs.to(cuda_0)

            outputs = model(inputs)
            all_outputs.append(outputs.data.cpu())

        all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    if use_cuda:
        all_labels = all_labels.to(cuda_0)
        all_outputs = all_outputs.to(cuda_0)

    return all_labels, all_outputs


def train(model, criterion, trainloader, testloader, optimizer, epochs, epoch_count=None,
          require_test = True,
          split=False, idx_s=0, idx_e=50000):
    model.train()
    for epoch in range(epochs):
        epoch += 1
        pbar = tqdm(trainloader, total=len(trainloader))
        train_loss_all = .0
        epoch_print = epoch if epoch_count is None else epoch_count
        for batch_id, (inputs, labels) in enumerate(pbar):
            if split:
                if batch_id == 0:
                    skip = 0
                    batch_size = len(inputs)
                idx = batch_id * batch_size
                if idx < idx_s or idx >= idx_e: skip+=1; continue
            if use_cuda:
                inputs = inputs.to(cuda_0)
                labels = labels.to(cuda_0)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_all += loss.data
            if split:
                batch_id -= skip
            train_loss = train_loss_all/(batch_id+1)
            pbar.set_description(f'Epoch: {epoch_print} - loss: {train_loss:5f}.')

        if require_test:
            labels, outputs = predict(model, testloader)
            _, preds = torch.max(outputs.data, dim=1)
            error_test = torch.mean((preds!=labels.data).float()).data
            labels, outputs = predict(model, trainloader, split=split, idx_s=idx_s, idx_e=idx_e)
            _, preds = torch.max(outputs.data, dim=1)
            error_train = torch.mean((preds!=labels.data).float()).data
            print(f'train_acc: {1-error_train:5f} - val_acc: {1-error_test:5f}')


    if require_test:
        return error_train, error_test, train_loss
    else:
        return False, False, train_loss


def transfer(model, class_num, randseed=233, init=nn.init.xavier_uniform_):
    fan_in = next(model.linear.parameters()).shape[-1]
    model.linear = nn.Linear(fan_in, class_num)
    torch.manual_seed(randseed)
    init(next(model.linear.parameters()))


def write_csv(error_train, error_test, loss, filename):
    if not len(error_train) == len(error_test) == len(loss):
        raise ValueError('Length of inputs do not match.')
    else:
        epoch = range(len(error_train))
    csvfile = open(filename, 'w')
    fieldnames = ['epoch', 'error_train', 'error_test', 'loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(error_train)):
        writer.writerow({'epoch': epoch[i],
                         'error_train': error_train[i],
                         'error_test': error_test[i],
                         'loss': loss[i]})


def tensor_to_generator(tensor):
    i = 0
    while i < len(tensor):
        yield tensor[i:i+1]
        i += 1


def dataloader_to_tensor(dataloader, get_label=False):
    images = torch.Tensor()
    labels = torch.Tensor()
    if use_cuda:
        images = images.to(cuda_0)
        labels = labels.to(cuda_0)
    for inputs, targets in dataloader:
        if use_cuda:
            inputs = inputs.to(cuda_0)
            if get_label:
                targets = targets.to(cuda_0)
        images = torch.cat((images, inputs), dim=0)
        if get_label:
            labels = torch.cat((labels, targets), dim=0)
    if use_cuda:
        images = images.cpu()
        labels = labels.cpu()

    return images, labels


def visualize(imagetensor, image_idx=1, filename='/tmp/tmp.png'):
    images = imagetensor.permute(0,2,3,1).numpy()
    imsave(filename, images[image_idx], format='png')
