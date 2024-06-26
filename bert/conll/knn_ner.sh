export PYTHONPATH="$PWD"

DARA_DIR="/home/xx/ner/ner_data/en_data/conll-2003"
FILE_NAME="word.bmes"
SAVE_PATH="/home/xx/ner/en_result/conll_bert_large"
BERT_PATH="/home/xx/ner/models/bert-large-cased"
PARAMS_FILE="/home/xx/ner/en_result/conll_bert_large/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/home/xx/ner/en_result/conll_bert_large/checkpoint/epoch=2_v2.ckpt"
DATASTORE_PATH="/home/xx/ner/ner_data/en_data/conll-2003/train-datastore-large"
link_temperature=0.001
link_ratio=0.10
topk=512

CUDA_VISIBLE_DEVICES=1 python  ../knn_ner_trainer.py \
--bert_path $BERT_PATH \
--batch_size 2 \
--workers 16 \
--max_length 512 \
--data_dir $DARA_DIR \
--file_name $FILE_NAME \
--path_to_model_hparams_file $PARAMS_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--datastore_path $DATASTORE_PATH \
--save_path $SAVE_PATH \
--link_temperature $link_temperature \
--link_ratio $link_ratio \
--topk $topk \
--gpus="1"
