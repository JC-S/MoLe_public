from model import *
from mole_utils import *
import time
import random as rd
from parameters import *


def gen_d2r_conv(save=False):
    print('Building Network..')
    model = VGG('VGG16', dropout=dropout, num_classes=100 if use_cifar100 else 10)
    print('Loading pretrained weight..')
    model.load_state_dict(torch.load(pretrain_weight))
    print('Finished loading.')
    conv_kernel = model.first_layer.weight.detach()
    kernel_size = conv_kernel.shape[-1]
    channel_out = conv_kernel.shape[0]
    channel_in = conv_kernel.shape[1]
    C_r_row = image_size**2*channel_in
    C_r_column = output_size**2*channel_out
    C_r = torch.zeros(C_r_row, C_r_column)
    m_square = image_size**2
    n_square = output_size**2

    print('Start composing C_r.')
    time_s = time.time()

    # For some reasons, numpy array works way faster here than pytorch tensors.
    # Convert to numpy and then convert back to tensor for better performance
    use_numpy = True
    if use_numpy:
        C_r = C_r.numpy()
        conv_kernel = conv_kernel.numpy()


    for ch_out_idx in range(channel_out):
        for in_ro_idx in range(image_size):
            for ch_in_idx in range(channel_in):
                for in_co_idx in range(image_size):
                    if in_co_idx < image_size-(kernel_size-1) and in_ro_idx < image_size-(kernel_size-1):
                        y_at_ch_out = in_co_idx + in_ro_idx * output_size
                        y = y_at_ch_out + ch_out_idx*n_square
                        x_at_ch_in = in_co_idx + in_ro_idx * image_size
                        x = x_at_ch_in + ch_in_idx*m_square
                        for k_ro_idx in range(kernel_size):
                            x1 = x + k_ro_idx * image_size
                            C_r[x1:x1+kernel_size, y] = conv_kernel[ch_out_idx, ch_in_idx, k_ro_idx]


    if use_numpy:
        C_r = torch.from_numpy(C_r)
        conv_kernel = torch.from_numpy(conv_kernel)


    time_e = time.time()
    print(f'Finished composing C_r. Time elapsed: {time_e-time_s: .2f} seconds.')

    if save:
        torch.save(C_r, save)

    return C_r, model, (channel_out, channel_in, kernel_size)


def sanity_check(C_r, model, layer_detail=()):
    trainloader, testloader = cifar_dataset(1, 0, shuffle_train=False, cifar100=use_cifar100)
    channel_out, channel_in, kernel_size = layer_detail
    for input, label in trainloader:
        original_output = model.first_layer(input)
        input_padded = torch.zeros(1, 3, image_size, image_size)
        input_padded[:,:, image_padding:image_size-image_padding, image_padding:image_size-image_padding] = input
        C_r_output = torch.mm(input_padded.view(1, image_size**2*channel_in), C_r).view(1, channel_out, output_size, output_size)
        break
    difference = C_r_output - original_output
    if difference.norm() < channel_out*output_size**2*1e-4:
        print(f'Sanity check passed. L2 distance is: {difference.norm()}.')


def comb_M_inverse(M_inverse, C_r):
    if use_cuda:
        M_inverse = M_inverse.to(cuda_0)
        C_r = C_r.to(cuda_0)
    C_ac = torch.mm(M_inverse, C_r)
    if use_cuda:
        C_ac.cpu()
    return C_ac


def channel_randomazition(C_ac, channel_out):
    order = []
    for i in range(channel_out):
        order += [i]
    rd.shuffle(order)
    C_ac_new = torch.zeros_like(C_ac)
    n_square = output_size**2
    for i in range(channel_out):
        channel_old = order[i]
        C_ac_new[:, i*n_square:(i+1)*n_square] = C_ac[:, channel_old*n_square: (channel_old+1)*n_square]
    return C_ac_new


if __name__ == '__main__':
    C_r, model, layer_detail = gen_d2r_conv(save='C_r.pt')
    sanity_check(C_r, model, layer_detail)
    M_inverse = torch.load('inverse_matrix.pt')
    C_ac = comb_M_inverse(M_inverse, C_r)
    C_ac = channel_randomazition(C_ac, layer_detail[0])
    torch.save(C_ac, 'aug-conv.pt')