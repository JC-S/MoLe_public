from model import *
from mole_utils import *
from parameters import *


class VGG_ac(nn.Module):
    def __init__(self, model_ori, C_ac):
        super(VGG_ac, self).__init__()
        self.C_ac = C_ac
        self.features = self._make_layers(model_ori)
        self.linear = nn.Linear(512, 100 if use_cifar100 else 10)
        self.n_input = C_ac.shape[0]
        self.channel_out = int(C_ac.shape[1] / output_size**2)

    def _make_layers(self, model_ori):
        layers = model_ori.layers
        return nn.Sequential(*layers)

    # def _padding(self, img):
    #     if image_padding:
    #         shape = img.shape
    #         shape[-1] += image_padding
    #         shape[-2] += image_padding
    #         img_padded = torch.zeros(shape)
    #         img_padded[:,:, image_padding:image_size-image_padding, image_padding:image_size-image_padding] = img
    #         return img_padded
    #     else:
    #         return img

    def forward(self, x):
        out = x.matmul(self.C_ac)
        out = out.view(out.shape[0], self.channel_out, output_size, output_size)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        return out


def main():
    lr = lr_ac
    print('Building original network...')
    model_ori = VGG('VGG16', dropout=dropout, num_classes=100 if use_cifar100 else 10)
    print('Finished building original network.')
    C_ac = torch.load('aug-conv.pt')
    morphing_matrix = torch.load('morphing_matrix.pt')
    if use_cuda:
        C_ac = C_ac.to(cuda_0)
        morphing_matrix = morphing_matrix.to(cuda_0)
    model = VGG_ac(model_ori, C_ac)
    trainloader, testloader = cifar_dataset(batch_size, num_workers_ac, shuffle_train, cifar100=use_cifar100,
                                            data_morphing=True, morph_matrix=morphing_matrix)
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
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=weight_decay*.1)
        _, error, loss = train(model, criterion, trainloader, testloader, optimizer, 1, epoch_count=epoch+1)
        acc = 1 - error
        if acc > best_acc:
            best_acc = acc
            best_loss = loss
            torch.save(model.state_dict(), train_ac_weight)
    print('-'*140)
    print(f'Best model stat: Loss = {best_loss:.5f}, Error = {(1-best_acc):.5f}')

if __name__ == '__main__':
    main()