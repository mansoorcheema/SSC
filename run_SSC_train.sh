#!/usr/bin/env bash


source venv/bin/activate


python ./main.py \
--model='palnet' \
--dataset=nyu \
--epochs=50 \
--batch_size=1 \
--workers=1 \
--lr=0.01 \
--lr_adj_n=10 \
--lr_adj_rate=0.1 \
--model_name='SSC_PALNet' 2>&1 |tee train_PALNet_NYU_010.log


deactivate

