export PYTHONPATH="$PWD"

DARA_DIR="/home/xg/ner/ner_data/zh_data/ontonote4"
FILE_NAME="char.bmes"
SAVE_PATH="/home/xg/ner/chinese_bert_result/ontonotes_base"
BERT_PATH="/home/xg/ner/models/ChineseBERT-base"
PARAMS_FILE="/home/xg/ner/chinese_bert_result/chinese_base_ontonotes/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/home/xg/ner/chinese_bert_result/chinese_base_ontonotes/checkpoint/epoch=4.ckpt"
DATASTORE_PATH="/home/xg/ner/ner_data/zh_data/ontonote4/train-datastore-chineseBERT-base"

CUDA_VISIBLE_DEVICES=0 python ../../build_datastore.py \
--bert_path $BERT_PATH \
--batch_size 10 \
--workers 4 \
--max_length 512 \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--path_to_model_hparams_file $PARAMS_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--datastore_path $DATASTORE_PATH \
--gpus="1"
