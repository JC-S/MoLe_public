from mole_utils import *
from parameters import *


def gen_morphing_matrix():
    morphing_size_max = image_size**2*3
    morphing_size = round(morphing_size_max / kappa)
    morphing_matrix = torch.rand(morphing_size, morphing_size).type(torch.float64)
    if kappa != 1:
        morphing_matrix_full = torch.zeros(morphing_size_max, morphing_size_max).type(torch.float64)
        for i in range(kappa):
            morphing_matrix_full[i*morphing_size:(i+1)*morphing_size, i*morphing_size:(i+1)*morphing_size] = morphing_matrix
        morphing_matrix = morphing_matrix_full
    morphing_matrix = morphing_matrix.type(torch.float64)
    if use_cuda:
        morphing_matrix = morphing_matrix.to(cuda_0)
    inverse_matrix = torch.inverse(morphing_matrix)
    morphing_matrix = morphing_matrix.type(torch.float32)
    inverse_matrix = inverse_matrix.type(torch.float32)
    torch.save(morphing_matrix.cpu(), 'morphing_matrix.pt')
    torch.save(inverse_matrix.cpu(), 'inverse_matrix.pt')


def sanity_check():
    morphing_matrix = torch.load('morphing_matrix.pt')
    inverse_matrix = torch.load('inverse_matrix.pt')
    trainloader, testloader = cifar_dataset(1, 0, shuffle_train=False, cifar100=use_cifar100)
    for input, label in trainloader:
        input = input.view(input.shape[1],input.shape[2],input.shape[3])
        input_padded = data_padding(input)
        input_r = input_padded.view(1, image_size**2*3)
        input_recover = input_r.mm(morphing_matrix).mm(inverse_matrix)
        input_recover = input_recover.view(3, image_size, image_size)
        difference = (input_padded - input_recover).norm()
        print(f'Difference is: {difference}.')
        break


if __name__ == '__main__':
    gen_morphing_matrix()
    sanity_check()
