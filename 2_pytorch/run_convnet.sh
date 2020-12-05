#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 2 \
    --hidden-dim 6 \
    --epochs 10 \
    --weight-decay 0.3 \
    --momentum 0.82 \
    --batch-size 512 \
    --lr 0.01 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
