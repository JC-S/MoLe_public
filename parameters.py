# glocal settings
use_cifar100 = False
image_padding = 1
image_size  = 32+image_padding*2

# pretrain parameters
pretrain_lr = 0.001
batch_size = 64
num_workers = 4
shuffle_train = True
pretrain_epochs = 300
decay_epochs = 60
dropout = 0.5
weight_decay = 5e-6
pretrain_weight = 'pretrain_weight_cifar100.pt' if use_cifar100 else 'pretrain_weight_cifar10.pt'

# data morphing settings
kappa = 1
output_size = 32

# train_with_ac parameters
lr_ac = pretrain_lr
train_ac_weight = 'train_ac_weight.pt'
train_noac_weight = 'train_noac_weight.pt'
num_workers_ac = 0 # Using multi-threading for GPU pre-processing will actually hurt performance.
