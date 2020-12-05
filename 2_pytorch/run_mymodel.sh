#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 10 \
    --epochs 15 \
    --weight-decay 0.3 \
    --momentum 0.85 \
    --batch-size 300 \
    --lr 0.01 | tee mymodel.log

#python -u test.py \
 #   --model convnet.pt\
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

