# Morphed Learning
This is an anonymous repo for NeurIPS-2019 submisssion:
"Towards Efficient and Secure Delivery of Data for Training and Inference with Privacy-Preserving"

Link to the preprint: [preprint](https://arxiv.org/abs/1809.09968)
(the preprint may not always be up-to-date).

### Abstract:
Privacy recently emerges as a severe concern in deep learning, e.g., sensitive data must be prohibited from being shared to the third party during deep neural network development.
In this paper, we present Morphed Learning (MoLe), an efficient and secure scheme for delivering deep learning data. 
MoLe is consisted of data morphing and Augmented Convolutional (Aug-Conv) layer.
Data morphing allows the data provider to send morphed data without privacy information, while Aug-Conv layer helps the deep learning developer apply their network on the morphed data without performance penalty.
Theoretical analysis show that MoLe can provide strong security with overhead non-related to dataset size or the depth of neural network.
Thanks to the low overhead, MoLe can be applied for both training and inference stages.
Specifically, using MoLe for VGG-16 network on CIFAR dataset, the computational overhead is 9\% and data transmission overhead is 5.12\%.
Meanwhile the attack success probability for the adversary is less than 7.9 x 10^{-90}.

# How to use:
## Step 1: Install dependency

You can find the dependent package in requirements.txt

It is recommended to use Python 3.7.

## Step 2: Pretrain a network that works on plain dataset

`python pretrain.py`

## Step 3: Generate morphing matrix and its inverse matrix

`python gen_morphing_matrix.py`

## Step 4: Compose the Aug-Conv layer

`python gen_augconv.py`

## Step 5: Test the effectiveness of Aug-Conv layer

Train the network with Aug-Conv layer on morphed CIFAR dataset:

`python train_with_ac.py`

This should give a test accuracy close to `pretrain.py`.

Train the network without Aug-Conv layer on morphed CIFAR dataset:

`python train_no_ac.py`

This should give a very low accuracy.

# Hyperparameters:

All hyperparameters can be found in `parameter.py`.