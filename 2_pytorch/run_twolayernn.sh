#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model twolayernn \
    --hidden-dim 600 \
    --epochs 6 \
    --weight-decay 0.1 \
    --momentum 0.85 \
    --batch-size 700 \
    --lr 0.006 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
