import os
import re
import json
import argparse
import logging
from functools import partial
from collections import namedtuple

from torch import nn

import numpy as np
from ner_trainer import NERTask
from utils import collate_to_max_length
from dataset import NERDataset
from utils import set_random_seed
from metrics import SpanF1ForNER

# enable reproducibility
# https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
set_random_seed(2333)

import torch
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup, BertForTokenClassification, RobertaForTokenClassification, RobertaConfig

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# 大致理解了，LightningModule定义的是一个系统而不是单纯的网络架构
class RatioNERTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a model, tokenizer and config"""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        # 初始化model、config等
        self.en_roberta = args.en_roberta
        # self.en_roberta = False
        self.entity_labels = NERDataset.get_labels(os.path.join(args.data_dir, "ner_labels.txt"))
        self.bert_dir = args.bert_path
        self.num_labels = len(self.entity_labels)
        if not self.en_roberta:
            self.bert_config = BertConfig.from_pretrained(self.bert_dir, output_hidden_states=True,
                                                          return_dict=True, num_labels=self.num_labels,
                                                          hidden_dropout_prob=self.args.hidden_dropout_prob)
            self.model = BertForTokenClassification.from_pretrained(self.bert_dir, config=self.bert_config)
        else:
            self.bert_config = RobertaConfig.from_pretrained(self.bert_dir, output_hidden_states=True,
                                                             return_dict=True, num_labels=self.num_labels,
                                                             hidden_dropout_prob=self.args.hidden_dropout_prob)
            self.model = RobertaForTokenClassification.from_pretrained(self.bert_dir, config=self.bert_config)

        for param in self.model.parameters():
            param.requires_grad = False
        self.custom_parameter = nn.Parameter(torch.zeros(1))

        # 初始化评估方法，抽象为类属性
        self.ner_evaluation_metric = SpanF1ForNER(entity_labels=self.entity_labels, save_prediction=self.args.save_ner_prediction)
        self.num_gpus = self.args.gpus

        info = json.load(open(os.path.join(self.args.datastore_path, "datastore_info.json")))
        key_file = os.path.join(self.args.datastore_path, "keys.npy")
        keys = np.memmap(key_file,
                         dtype=np.float32,
                         mode="r",
                         shape=(info['token_sum'], info['hidden_size']))
        keys_in_memory = np.zeros((info['token_sum'], info['hidden_size']), dtype=np.float32)
        keys_in_memory[:] = keys[:]

        self.keys = torch.from_numpy(keys_in_memory)

        val_file = os.path.join(self.args.datastore_path, "vals.npy")
        vals = np.memmap(val_file,
                         dtype=np.int32,
                         mode="r",
                         shape=(info['token_sum'],))
        vals_in_memory = np.zeros((info['token_sum'],), dtype=np.int64)
        vals_in_memory[:] = vals[:]

        self.vals = torch.from_numpy(vals_in_memory)

        self.link_temperature = torch.tensor(self.args.link_temperature)

        self.link_ratio = torch.tensor(self.args.link_ratio)

        if (self.num_gpus):
            # self.keys = self.keys.transpose(0, 1).cuda() # [feature_size, token_num]
            # self.norm_1 = (self.keys ** 2).sum(dim=0, keepdim=True).sqrt() # [1, token_num]
            self.keys = self.keys.cuda()
            self.vals = self.vals.cuda()
            self.link_temperature = self.link_temperature.cuda()
            self.link_ratio = self.link_ratio.cuda()

        # format = '%(asctime)s - %(name)s - %(message)s'
        # logging.basicConfig(format=format, filename=os.path.join(self.args.save_path, "eval_result_log.txt"),
        #                     level=logging.INFO)

        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)


    # 设置optimizer
    def configure_optimizers(self):
        optimizer = AdamW([self.custom_parameter], lr=5e-1)  # 只优化custom_parameter参数
        return optimizer



    def forward(self, input_ids):
        attention_mask = (input_ids != 0).long()
        outputs = self.model(input_ids, attention_mask=attention_mask)
        probabilities, argmax_labels = self.postprocess_logits_to_labels(outputs.logits,outputs.hidden_states[-1])

        outputs.logits = probabilities
        return outputs



    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        loss_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        bert_classifiaction_outputs = self.forward(input_ids=input_ids)
        loss = self.compute_loss(bert_classifiaction_outputs.logits, labels, loss_mask=loss_mask)

        tf_board_logs = {
            "train_loss": loss,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"]
        }
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, gold_labels = batch
        batch_size, seq_len = input_ids.shape
        loss_mask = (input_ids != 0).long()
        bert_classifiaction_outputs = self.forward(input_ids=input_ids)
        loss = self.compute_loss(bert_classifiaction_outputs.logits, gold_labels, loss_mask=loss_mask)
        probabilities, argmax_labels = self.postprocess_logits_to_labels(bert_classifiaction_outputs.logits.view(batch_size, seq_len, -1),bert_classifiaction_outputs.hidden_states[-1])
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=loss_mask)
        return {"val_loss": loss, "confusion_matrix": confusion_matrix}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> valid_f1 is: {f1}")
        self.result_logger.info(f"EVAL INFO -> ratio is: {torch.sigmoid(self.custom_parameter)}")
        tensorboard_logs = {"val_loss": avg_loss, "val_f1": f1, }
        return {"val_loss": avg_loss, "val_log": tensorboard_logs, "val_f1": f1, "val_precision": precision, "val_recall": recall}

    def train_dataloader(self,) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self, ) -> DataLoader:
        return self.get_dataloader("dev")

    def _load_dataset(self, prefix="test"):
        vocab_file = self.args.bert_path
        if not self.en_roberta:
            vocab_file = os.path.join(self.args.bert_path, "vocab.txt")
        dataset = NERDataset(directory=self.args.data_dir, prefix=prefix,
                                      vocab_file=vocab_file,
                                      max_length=self.args.max_length,
                                      config_path=os.path.join(self.args.bert_path, "config"),
                                      file_name=self.args.file_name, lower_case=self.args.lower_case,
                                      language=self.args.language, en_roberta=self.en_roberta)

        return dataset

    def get_dataloader(self, prefix="train", limit=None) -> DataLoader:
        """return {train/dev/test} dataloader"""
        dataset = self._load_dataset(prefix=prefix)

        if prefix == "train":
            batch_size = self.args.train_batch_size
            # small dataset like weibo ner, define data_generator will help experiment reproducibility.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            batch_size = self.args.eval_batch_size
            data_sampler = SequentialSampler(dataset)

        # sampler option is mutually exclusive with shuffle
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size,
                                num_workers=self.args.workers, collate_fn=partial(collate_to_max_length, fill_values=[0, 0, 0]),
                                drop_last=False)

        return dataloader

    def test_dataloader(self, ) -> DataLoader:
        return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
        input_ids, gold_labels = batch
        sequence_mask = (input_ids != 0).long()
        batch_size, seq_len = input_ids.shape
        bert_classifiaction_outputs = self.forward(input_ids=input_ids)
        probabilities, argmax_labels = self.postprocess_logits_to_labels(bert_classifiaction_outputs.logits.view(batch_size, seq_len, -1),
            bert_classifiaction_outputs.hidden_states[-1])
        confusion_matrix = self.ner_evaluation_metric(argmax_labels, gold_labels, sequence_mask=sequence_mask)
        return {"confusion_matrix": confusion_matrix}

    def test_epoch_end(self, outputs):
        confusion_matrix = torch.stack([x["confusion_matrix"] for x in outputs]).sum(0)
        all_true_positive, all_false_positive, all_false_negative = confusion_matrix
        if self.args.save_ner_prediction:
            precision, recall, f1, entity_tuple = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative, prefix="test")
            gold_entity_lst, pred_entity_lst = entity_tuple
            self.save_predictions_to_file(gold_entity_lst, pred_entity_lst)
        else:
            precision, recall, f1 = self.ner_evaluation_metric.compute_f1_using_confusion_matrix(all_true_positive, all_false_positive, all_false_negative)

        tensorboard_logs = {"test_f1": f1}
        self.result_logger.info(f"TEST RESULT -> TEST F1: {f1}, Precision: {precision}, Recall: {recall} ")
        return {"test_log": tensorboard_logs, "test_f1": f1, "test_precision": precision, "test_recall": recall}


    def save_save_predictions_to_file(self, gold_entity_lst, pred_entity_lst, prefix="test"):
        dataset = self._load_dataset(prefix=prefix)
        data_items = dataset.data_items

        save_file_path = os.path.join(self.args.save_path, "test_predictions.txt")
        print(f"INFO -> write predictions to {save_file_path}")
        with open(save_file_path, "w") as f:
            for gold_label_item, pred_label_item, data_item in zip(gold_entity_lst, pred_entity_lst, data_items):
                data_tokens = data_item[0]
                f.write("=!" * 20+"\n")
                f.write("".join(data_tokens)+"\n")
                f.write(gold_label_item+"\n")
                f.write(pred_label_item+"\n")
    def compute_loss(self, logits, labels, loss_mask=None):
        """
        Desc:
            compute cross entropy loss
        Args:
            logits: FloatTensor, shape of [batch_size, sequence_len, num_labels]
            labels: LongTensor, shape of [batch_size, sequence_len, num_labels]
            loss_mask: Optional[LongTensor], shape of [batch_size, sequence_len].
                1 for non-PAD tokens, 0 for PAD tokens.
        """
        loss_fct = CrossEntropyLoss()
        if loss_mask is not None:
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
    def postprocess_logits_to_labels(self, logits, hidden):
        """input logits should in the shape [batch_size, seq_len, num_labels]"""
        probabilities = F.softmax(logits, dim=2)  # shape of [batch_size, seq_len, num_labels]
        # 需要做的是：将中间的store中的相同的labels的向量加起来，取平均转换成一个更新的向量，这样就不用再去找topk的向量直接把keys进行合并，vals相同的合并加到一起，然后取平均。
        # 最后找相似的向量的时候，就根据余弦相似度直接得到一个概率值。
        # 第一步：将keys相同的vals合并起来并取平均，得到只有单一标签的keys并经过平均后的列表
        # 第二步：使用hidden向量与keys向量计算相似度得分，将相似度得分转换成概率scores_probabilities。
        # 第三步：将两个概率列表根据λ进行加和

        # 第一步：将keys相同的vals合并起来并取平均，得到只有单一标签的keys并经过平均后的列表
        # keys 是一个形状为 [token_num ,feature_size ] 的张量
        # vals 是一个形状为 [token_num] 的张量，表示每个样本的标签
        unique_labels, inverse_indices = torch.unique(self.vals, return_inverse=True)

        # 初始化聚合后的 keys 和标签数组
        merged_keys = torch.zeros(len(unique_labels), self.keys.size(1))
        # merged_vals = unique_labels

        # 对 keys 进行聚合和均值计算
        for i, label in enumerate(unique_labels):
            mask = inverse_indices == i
            merged_keys[i] = torch.mean(self.keys[mask], dim=0)

        # merged_vals = torch.tensor(merged_vals)  # 合并后的标签张量

        # 第二步：使用hidden向量与merged_keys向量计算相似度，根据对应的merged_vals转换成概率列表。
        # hidden = torch.tensor([[[0.3, 0.2, -0.5], [0.7, 0.2, -0.9]],
        #                        [[0.2, 0.3, -0.5], [0.5, 0.3, -0.8]]])  # [bsz , sent_len, feature_size]

        batch_size = hidden.shape[0]
        sent_len = hidden.shape[1]
        hidden_size = hidden.shape[-1]

        hidden = hidden.view(-1, hidden_size)  # [bsz*sent_len, feature_size]
        merged_keys = merged_keys.transpose(0, 1).cuda()  # [feature_size, num_labels]
        norm_1 = (merged_keys ** 2).sum(dim=0, keepdim=True).sqrt()  # [1, num_labels]

        # 计算测试token的模长
        norm_2 = (hidden ** 2).sum(dim=1, keepdim=True).sqrt()  # [bsz*sent_len, 1]
        sim = torch.mm(hidden, merged_keys)  # [bsz*sent_len, num_labels]
        # 计算merged_keys和hidden的余弦相似度
        scores = (sim / (norm_1 + 1e-10) / (norm_2 + 1e-10)).view(batch_size, sent_len,
                                                                  -1)  # [bsz, sent_len, num_labels]

        # 将scores转换成概率列表，设为probabilities向量
        scores_probabilities = torch.softmax(scores / self.link_temperature, dim=-1)  # [bsz, sent_len, num_labels]

        # ratio = torch.sigmoid(self.custom_parameter).cuda()
        ratio = self.custom_parameter.cuda()

        sim_mean = torch.mean(sim,dim=-1).view(batch_size, sent_len)

        ratio = ratio * sim_mean
        ratio_sigmod = torch.sigmoid(ratio).cuda()
        # print(sim_mean)
        # print(sim_mean.shape)
        # print(ratio)
        # print(ratio.shape)
        # print(ratio_sigmod.shape)
        # print(ratio_sigmod)
        # print(1-ratio_sigmod)
        # exit(0)

        # print(ratio_sigmod.shape)
        # 使用unsqueeze方法添加一个新的维度
        ratio_sigmod = ratio_sigmod.unsqueeze(-1)
        # 使用expand方法扩展张量的最后一维
        ratio_sigmod = ratio_sigmod.expand(batch_size, sent_len, self.num_labels)
        # print(ratio_sigmod.shape)
        # print(ratio_sigmod)
        # exit(0)
        probabilities = ratio_sigmod * scores_probabilities + (1 - ratio_sigmod) * probabilities
        # [bsz,sent_len]  [bsz, sent_len, num_labels]
        argmax_labels = torch.argmax(probabilities, 2, keepdim=False)  # [bsz, sent_len]
        return probabilities,argmax_labels

    def on_save_checkpoint(self, checkpoint):
        # 在检查点保存时将参数ratio保存到checkpoint中
        checkpoint['custom_parameter'] = self.custom_parameter.tolist()  # 使用tolist()将Tensor转换为Python列表

    def on_load_checkpoint(self, checkpoint):
        # 在加载检查点时从checkpoint中恢复参数ratio的值
        self.custom_parameter = torch.tensor(checkpoint['custom_parameter'])  # 将Python列表转换为Tensor
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", type=str, help="bert config file")
    parser.add_argument("--train_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", type=str, help="train data path")
    parser.add_argument("--save_path", type=str, help="train data path")
    parser.add_argument("--save_topk", default=1, type=int, help="save topk checkpoint")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, )
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--classifier", type=str, default="single")
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--file_name", default="", type=str, help="use for truncated sets.")
    parser.add_argument("--save_ner_prediction", action="store_true", help="only work for test.")
    parser.add_argument("--path_to_model_hparams_file", default="", type=str, help="use for evaluation")
    parser.add_argument("--checkpoint_path", default="", type=str, help="use for evaluation.")
    parser.add_argument("--lower_case", default=False, type=bool, help="lowercase when load English data.")
    parser.add_argument("--language", default="en", type=str, help="the language of the dataset.")
    parser.add_argument("--en_roberta", action="store_true", help="whether load roberta for classification or not.")

    parser.add_argument("--datastore_path", default="", type=str, help="use for saving datastore.")
    parser.add_argument("--link_temperature", default=1.0, type=float, help="temperature used by edge linking.")
    parser.add_argument("--link_ratio", default=0.0, type=float, help="ratio of vocab probs predicted by edge linking.")

    return parser


def main():
    # 得到ArgumentParser()对象，然后将该命令行解析器添加到与训练器相关的参数中去
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # 用的还是之前已经训练好的NER模型
    print("-----------")
    print(args.checkpoint_path)
    ner_model = NERTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                             hparams_file=args.path_to_model_hparams_file,
                                             map_location=None)

    # 初始化model
    model = RatioNERTask(args)
    model.model = ner_model.model
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_path, "ratio_checkpoint", "{epoch}",),
        save_top_k=args.save_topk,
        save_last=False,
        monitor="val_f1",
        mode="max",
        verbose=True,
        period=-1,
    )

    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log')

    # save args
    with open(os.path.join(args.save_path, "ratio_checkpoint", "args.json"), "w") as f:
        args_dict = args.__dict__
        del args_dict["tpu_cores"]
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger,
                                         deterministic=True)
    # trainer.fit(model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.save_path)
    print(path_to_best_checkpoint)
    print("-----------------@@@@@@@@@@")
    checkpoint = torch.load(path_to_best_checkpoint)
    model.load_state_dict(checkpoint)
    model.result_logger.info("=&"*20)
    model.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    model.result_logger.info(f"Best ratio on DEV set is {checkpoint['custom_parameter']}")
    trainer.test(model)
    model.result_logger.info("=&"*20)


def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log_true_v_ratio_temperature.txt"):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN = re.compile(r"val_f1 reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/glyce/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_f1_on_dev = 0
    best_checkpoint_on_dev = 0
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("val_f1 reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(" as top", "")

        if current_f1 >= best_f1_on_dev:
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def evaluate():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = RatioNERTask.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                            hparams_file=args.path_to_model_hparams_file,
                                                            map_location=None,
                                                            batch_size=1)
    trainer = Trainer.from_argparse_args(args, deterministic=True)

    trainer.test(model)

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
    # evaluate()