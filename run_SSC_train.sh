#!/usr/bin/env bash


source venv/bin/activate


python ./main.py \
--model='palnet' \
--dataset=nyu \
--epochs=5 \
--batch_size=1 \
--workers=0 \
--lr=0.01 \
--lr_adj_n=1 \
--lr_adj_rate=0.1 \
--model_name='SSC_PalNet' 2>&1 |tee train_PalNet_NYU.log


deactivate

