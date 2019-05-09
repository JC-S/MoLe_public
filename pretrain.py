from model import *
from mole_utils import *
from parameters import *


def pretrain():
    lr = pretrain_lr
    print('Building network...')
    model = VGG('VGG16', dropout=dropout, num_classes=100 if use_cifar100 else 10)
    print('Finished building network.')
    trainloader, testloader = cifar_dataset(batch_size, num_workers, shuffle_train, cifar100=use_cifar100)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        model = model.to(cuda_0)
    print('Start Training...')
    print(f'Init lr is: {lr: .4f}')
    for epoch in range(pretrain_epochs):
        best_acc = 0
        best_loss = 0
        lr_decay = epoch % decay_epochs == 0 and epoch != 0
        if lr_decay:
            lr *= 0.5
            print(f'Set lr to {lr: .4f}')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=weight_decay)
        _, error, loss = train(model, criterion, trainloader, testloader, optimizer, 1, epoch_count=epoch+1)
        acc = 1 - error
        if acc > best_acc:
            best_acc = acc
            best_loss = loss
            torch.save(model.state_dict(), pretrain_weight)
    print('-'*140)
    print(f'Best model stat: Loss = {best_loss:.5f}, Error = {(1-best_acc):.5f}')


if __name__ == '__main__':
    pretrain()
