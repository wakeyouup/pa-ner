export PYTHONPATH="$PWD"

DARA_DIR="/home/xx/ner/ner_data/en_data/conll-2003"
FILE_NAME="word.bmes"
SAVE_PATH="/home/xx/ner/en_result/conll_bert_large"
BERT_PATH="/home/xx/ner/models/bert-large-cased"

CUDA_VISIBLE_DEVICES=1 python ../ner_trainer.py \
--lr 5e-5 \
--max_epochs 15 \
--max_length 512 \
--weight_decay 0.01 \
--hidden_dropout_prob 0.1 \
--warmup_proportion 0.001  \
--train_batch_size 16 \
--accumulate_grad_batches 2 \
--save_topk 20 \
--val_check_interval 0.25 \
--save_path $SAVE_PATH \
--bert_path $BERT_PATH \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--gpus="1"
