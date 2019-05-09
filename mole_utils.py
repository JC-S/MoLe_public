from utils import *
import torchvision
from torchvision import transforms
from parameters import *

def data_morphing(input_data, morph_matrix, reshape=False):
    """
    This function supports both batch and non-batch operation.
    Whether it is a batch operation depends on the shape of input_data.
    The shape of output_data is the same as input_data.
    """
    data_dim = len(input_data.shape)
    if data_dim not in [2, 3]:
        raise ValueError("Input_data should be 2-D or 3-D.")
    else:
        if image_padding:
            shape = list(input_data.shape)
            shape[-1] += image_padding*2
            shape[-2] += image_padding*2
            img_padded = torch.zeros(shape)
            img_padded[:, image_padding:image_size-image_padding, image_padding:image_size-image_padding] = input_data
            input_data = img_padded
        input_data = input_data.view(1, 3*image_size**2)
        if use_cuda:
            input_data = input_data.to(cuda_0)
        out = torch.matmul(input_data, morph_matrix)
        if reshape:
            out = out.view(shape)
            if image_padding:
                shape[-1] -= image_padding*2
                shape[-2] -= image_padding*2
                out_nopad = out[:, image_padding:image_size-image_padding, image_padding:image_size-image_padding]
                out = out_nopad
        return out


class data_morph_transpose(object):
    def __init__(self, morph_matrix, reshape=False):
        if len(morph_matrix.shape) != 2:
            raise ValueError("Morph matrix must be a 2-D tensor.")
        self.morph_matrix = morph_matrix
        self.reshape = reshape

    def __call__(self, img):
        return data_morphing(img, self.morph_matrix, self.reshape)


def cifar_dataset(batch_size=128, num_workers=2, download=True, shuffle_train=False, cifar100=False,
                  data_morphing=False, morph_matrix=[], reshape = False):
    data_transform_basic = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
    ]

    data_transform_train = data_transform_basic

    if data_morphing:
        data_transform_basic = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
            data_morph_transpose(morph_matrix, reshape),
        ]
        data_transform_train = data_transform_basic
        #data_transform_train = [
        #    transforms.RandomCrop(32, padding=4),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
        #    data_morph_transpose(morph_matrix),
        #]

    transform_train = transforms.Compose(data_transform_train)

    transform_test = transforms.Compose(data_transform_basic)

    if cifar100:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=download, transform=transform_train)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)

    if cifar100:
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=download, transform=transform_test)
    else:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def data_padding(data, padding=image_padding):
    shape = list(data.shape)
    if padding:
        shape[-1] += padding*2
        shape[-2] += padding*2
    data_padded = torch.zeros(shape)
    data_padded[:, padding:image_size-padding, padding:image_size-padding] = data
    return data_padded
