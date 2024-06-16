import os
import json
import argparse
import logging
from functools import partial

from utils import collate_to_max_length
from dataset import NERDataset
from utils import set_random_seed
from metrics import SpanF1ForNER
from ner_trainer import NERTask

set_random_seed(2333)

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader, SequentialSampler
from transformers import BertConfig, RobertaConfig

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


class KNNNERTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args

        self.en_roberta = args.en_roberta
        self.entity_labels = NERDataset.get_labels(os.path.join(args.data_dir, "ner_labels.txt"))
        self.bert_dir = args.bert_path
        self.num_labels = len(self.entity_labels)
        if not self.en_roberta:
            self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=False,
                                                          num_labels=self.num_labels,
                                                          hidden_dropout_prob=self.args.hidden_dropout_prob)
        else:
            self.bert_config = RobertaConfig.from_pretrained(self.bert_dir, output_hidden_states=False,
                                                             num_labels=self.num_labels,
                                                             hidden_dropout_prob=self.args.hidden_dropout_prob)
        self.model = None

        self.ner_evaluation_metric = SpanF1ForNER(entity_labels=self.entity_labels,
                                                  save_prediction=self.args.save_ner_prediction)

        self.num_gpus = self.args.gpus

        format = '%(asctime)s - %(name)s - %(message)s'
        logging.basicConfig(format=format, filename=os.path.join(self.args.save_path, "knn_result_log.txt"),
                            level=logging.INFO)
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, attention_mask=attention_mask)

    def test_dataloader(self, ) -> DataLoader:
        vocab_file = self.args.bert_path
        if not self.en_roberta:
            vocab_file = os.path.join(self.args.bert_path, "vocab.txt")

        dataset = NERDataset(directory=self.args.data_dir, prefix="test",
                             vocab_file=vocab_file,
                             max_length=self.args.max_length,
                             config_path=os.path.join(self.args.bert_path, "config"),
                             file_name=self.args.file_name, en_roberta=self.en_roberta)

        batch_size = self.args.batch_size
        data_sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers,
                                collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                                drop_last=False)

        info = json.load(open(os.path.join(self.args.datastore_path, "datastore_info.json")))
        key_file = os.path.join(self.args.datastore_path, "keys.npy")
        keys = np.memmap(key_file,
                         dtype=np.float32,
                         mode="r",
                         shape=(info['token_sum'], info['hidden_size']))
        keys_in_memory = np.zeros((info['token_sum'], info['hidden_size']), dtype=np.float32)  # 创建和keys相同的np数组
        keys_in_memory[:] = keys[:]

        val_file = os.path.join(self.args.datastore_path, "vals.npy")
        vals = np.memmap(val_file,
                         dtype=np.int32,
                         mode="r",
                         shape=(info['token_sum'],))
        vals_in_memory = np.zeros((info['token_sum'],), dtype=np.int64)
        vals_in_memory[:] = vals[:]

        keys_in_prototypical = np.zeros((self.num_labels, info['hidden_size']), dtype=np.float32)
        for i in range(self.num_labels):
            num_same_label = 0
            for index, value in np.ndenumerate(vals):
                if value == i:
                    num_same_label = num_same_label + 1
                    keys_in_prototypical[i] += keys_in_memory[index]
            keys_in_prototypical[i] = keys_in_prototypical[i] / num_same_label

        self.keys = torch.from_numpy(keys_in_prototypical)

        self.link_temperature = torch.tensor(self.args.link_temperature)

        self.link_ratio = torch.tensor(self.args.link_ratio)

        if (self.num_gpus):
            self.keys = self.keys.transpose(0, 1).cuda()  # [feature_size, label_num],转置一下
            self.norm_1 = (self.keys ** 2).sum(dim=0, keepdim=True).sqrt()  # [1, token_num]   keys向量的模长
            # self.vals = self.vals.cuda()
            self.link_temperature = self.link_temperature.cuda()
            self.link_ratio = self.link_ratio.cuda()

        return dataloader

    def test_step(self, batch, batch_idx):
        input_ids, gold_labels = batch
        sequence_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape

        bert_classifiaction_outputs = self.forward(input_ids=input_ids)

        # 差别就只在这一步，在测试的时候，根据训练好的Bert-NER模型输出的结果要综合kNN中key-value的结果
        # 其中bert_classifiaction_outputs.logits是[batch_size, seq_len, num_labels],还未经过softmax
        # bert_classifiaction_outputs.hidden_states是原始的没有拼接输出层的BertModel的outputs.hidden_states,也就是
        # [batch_size, seq_len, bert_feature]，outputs[0]==outputs.last_hidden_state
        # outputs.hidden_states和outputs.last_hidden_state不一样的。
        # hidden_states是由每一层的隐藏状态组成的元组，最后一层即是last_hidden_state，即last_hidden_state=hidden_states[-1]
        # 在这里是没办法直接得到last_hidden_state，故由hidden_states[-1]得到
        # 传进去的两个参数，一个是num_labels；一个是hidden_size
        argmax_labels = self.postprocess_logits_to_labels(bert_classifiaction_outputs.logits,
                                                          bert_classifiaction_outputs.hidden_states[-1])
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=sequence_mask)
        return {"confusion_matrix": confusion_matrix}

    def test_epoch_end(self, outputs):
        confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        if self.args.save_ner_prediction:
            precision, recall, f1, entity_tuple = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(
                all_true_positive, all_false_positive, all_false_negative, prefix="test")
            gold_entity_lst, pred_entity_lst = entity_tuple
            self.save_predictions_to_file(gold_entity_lst, pred_entity_lst)
        else:
            precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive,
                                                                                                 all_false_positive,
                                                                                                 all_false_negative)

        tensorboard_logs = {"test_f1": f1}
        self.result_logger.info(
            f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} , link_temperature: {self.link_temperature}, link_ratio: {self.link_ratio}")
        return {"test_log": tensorboard_logs, "test_f1": f1, "test_precision": precision, "test_recall": recall}

    def postprocess_logits_to_labels(self, logits, hidden):
        """input logits should in the shape [batch_size, seq_len, num_labels]"""
        probabilities = F.softmax(logits, dim=2)  # shape of [batch_size, seq_len, num_labels],LM的输出

        batch_size = hidden.shape[0]
        sent_len = hidden.shape[1]
        hidden_size = hidden.shape[-1]

        # cosine similarity，由测试的例子和datastore中的数据进行余弦相似度的计算
        hidden = hidden.view(-1, hidden_size)  # [bsz*sent_len, feature_size]
        # 画图示意，得到的最后一维上，每一组数据都表示input的与所有keys的内积
        sim = torch.mm(hidden, self.keys)  # [bsz*sent_len, label_num]
        # 计算测试token的feature的模长
        norm_2 = (hidden ** 2).sum(dim=1, keepdim=True).sqrt()  # [bsz*sent_len, 1]
        # 原型向量的模长已经在之前计算过了，这里直接拿来计算与每个原型向量的余弦相似度
        scores = (sim / (self.norm_1 + 1e-10) / (norm_2 + 1e-10)).view(batch_size, sent_len, -1)  # [bsz, sent_len, label_num]


        # 将余弦相似度经过softmax操作转化为概率，link_temperature则是控制器，控制概率的间隔的（参考知识蒸馏就知道）
        # 这里每个得分/link_temperature就已经是最终的knn输出的概率分布
        knn_probabilities = torch.softmax(scores / self.link_temperature, dim=-1)  # [bsz, sent_len, label_num]

        probabilities = self.link_ratio * knn_probabilities + (1 - self.link_ratio) * probabilities

        argmax_labels = torch.argmax(probabilities, 2, keepdim=False)  # [bsz, sent_len]
        return argmax_labels

    def save_save_predictions_to_file(self, gold_entity_lst, pred_entity_lst, prefix="test"):
        dataset = self._load_dataset(prefix=prefix)
        data_items = dataset.data_items

        save_file_path = os.path.join(self.args.save_path, "test_predictions.txt")
        print(f"INFO -> write predictions to {save_file_path}")
        with open(save_file_path, "w") as f:
            for gold_label_item, pred_label_item, data_item in zip(gold_entity_lst, pred_entity_lst, data_items):
                data_tokens = data_item[0]
                f.write("=!" * 20 + "\n")
                f.write("".join(data_tokens) + "\n")
                f.write(gold_label_item + "\n")
                f.write(pred_label_item + "\n")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", type=str, help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, )
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--file_name", default="", type=str, help="use for truncated sets.")
    parser.add_argument("--save_ner_prediction", action="store_true", help="only work for test.")
    parser.add_argument("--path_to_model_hparams_file", default="", type=str, help="use for evaluation")
    parser.add_argument("--checkpoint_path", default="", type=str, help="use for evaluation.")
    parser.add_argument("--datastore_path", default="", type=str, help="use for saving datastore.")
    parser.add_argument("--link_temperature", default=1.0, type=float, help="temperature used by edge linking.")
    parser.add_argument("--link_ratio", default=0.0, type=float, help="ratio of vocab probs predicted by edge linking.")
    parser.add_argument("--topk", default=64, type=int,
                        help="use topk-scored neighbor tgt nodes for link prediction and probability compuation.")
    parser.add_argument("--en_roberta", action="store_true", help="whether load roberta for classification or not.")

    return parser


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # 用的还是之前已经训练好的NER模型
    ner_model = NERTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                             hparams_file=args.path_to_model_hparams_file,
                                             map_location=None,
                                             batch_size=args.batch_size)

    model = KNNNERTask(args)
    model.model = ner_model.model

    logger = TensorBoardLogger(save_dir=args.save_path, name='knn')

    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=logger)

    # 唯一改变的就是测试部分，在测试时需要加上kNN部分的预测
    trainer.test(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()



