#!/usr/bin/env bash

source venv/bin/activate


python ./test.py \
--model='ddrnet' \
--dataset=nyu \
--batch_size=4 \
--resume='weights/003/cpBest_SSC_DDRNet.pth.tar' 2>&1 |tee test_DDRNet_NYU.log


deactivate

