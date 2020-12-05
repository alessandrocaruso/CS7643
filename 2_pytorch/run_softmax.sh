#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model softmax \
    --epochs 1 \
    --weight-decay 0.6 \
    --momentum 0.85 \
    --batch-size 200 \
    --lr 0.001 | tee softmax.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
