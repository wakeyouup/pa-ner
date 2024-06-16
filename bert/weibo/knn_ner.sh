export PYTHONPATH="$PWD"

DARA_DIR="/userhome/xx/ner/ner_data/weibo"
FILE_NAME="all.bmes"
SAVE_PATH="/userhome/xx/ner/chinese_result/weibo_bert_base"
BERT_PATH="/userhome/xx/ner/models/bert-base-chinese"
PARAMS_FILE="/userhome/xx/ner/chinese_result/weibo_bert_base/log/version_0/hparams.yaml"
CHECKPOINT_PATH="/userhome/xx/ner/chinese_result/weibo_bert_base/checkpoint/epoch=11_v0.ckpt"
DATASTORE_PATH="/userhome/xx/ner/ner_data/weibo/train-datastore-bert-base"
link_temperature=0.071
link_ratio=0.41
topk=-1

CUDA_VISIBLE_DEVICES=0 python ./knn_ner_trainer.py \
--bert_path $BERT_PATH \
--batch_size 1 \
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