#!/usr/bin/env bash


source venv/bin/activate


python ./main.py \
--model='ccpnet' \
--dataset=nyu \
--epochs=50 \
--batch_size=1 \
--workers=1 \
--lr=0.001 \
--lr_adj_n=25 \
--lr_adj_rate=0.1 \
--model_name='SSC_CCPNet' 2>&1 |tee train_CCPNet_NYU_009.log


deactivate

