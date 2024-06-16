export PYTHONPATH="$PWD"

DARA_DIR="/home/xg/ner/ner_data/zh_data/ontonote4"
FILE_NAME="char.bmes"
SAVE_PATH="/home/xg/ner/chinese_bert_result/chinese_base_ontonotes"
BERT_PATH="/home/xg/ner/models/ChineseBERT-base"

CUDA_VISIBLE_DEVICES=0 python ../../ner_trainer.py \
--lr 3e-5 \
--max_epochs 5 \
--max_length 275 \
--weight_decay 0.001 \
--hidden_dropout_prob 0.2 \
--warmup_proportion 0.1  \
--train_batch_size 26 \
--accumulate_grad_batches 1 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--precision=16 \
--optimizer torch.adam \
--classifier multi \
--gpus="1"
