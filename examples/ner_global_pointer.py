"""
Adapted from https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_efficient_global_pointer.py
"""
import os
import pathlib
from typing import List, Tuple, Dict

import bert4torch.layers
import bert4torch.losses
import bert4torch.snippets
import numpy as np
import torch
import torch.nn as nn
import transformers

import utils
import global_pointer_utils


class MyDataset(bert4torch.snippets.ListDataset):
    @staticmethod
    def load_data(filename) -> List[Tuple]:
        """
        File format

        text0char0 O
        text0char1 B-XX
        text0char2 I-XX
        text0char3 O
        text0char4 B-YY

        text1char0 ...
        ...
        """
        with open(filename, encoding='utf8') as f:
            lines = f.read()
        data = []
        for sample in lines.split('\n\n'):
            if not sample:
                continue
            # label [[first_idx, last_idx, entity], ...]
            text, label = '', []
            for i, line in enumerate(sample.split('\n')):
                if not line:
                    continue
                char, tag = line.rsplit(' ', maxsplit=1)
                text += char
                if tag[0] == 'B':
                    label.append([i, i, tag[len('B-'):]])
                elif tag[0] == 'I':
                    label[-1][1] = i
            data.append((text, label))
        return data


def make_collate_fn(tokenizer, max_length, label2id):
    def collate_fn(data):
        texts, pos_labels = list(zip(*data))
        batch_encoding = tokenizer(
            texts, max_length=max_length, truncation=True, padding='max_length',
        )

        batch_labels = []
        for i in range(len(texts)):
            first_mapping = {
                span[0]: i for i, span in enumerate(batch_encoding[i].offsets)
                if span != (0, 0)
            }
            last_mapping = {
                span[1] - 1: i for i, span in enumerate(batch_encoding[i].offsets)
                if span != (0, 0)
            }

            label_matrix = np.zeros((len(label2id), max_length, max_length))
            for first, last, label in pos_labels[i]:
                if first in first_mapping and last in last_mapping:
                    label_matrix[
                        label2id[label], first_mapping[first], last_mapping[last]
                    ] = 1
            batch_labels.append(label_matrix)

        final_dict = batch_encoding.data
        final_dict['labels'] = np.array(batch_labels)
        for k, v in final_dict.items():
            final_dict[k] = torch.LongTensor(v)

        return final_dict

    return collate_fn


class Model(nn.Module):
    def __init__(self, bert_path, num_tags, ner_hidden_size=64):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(bert_path)
        self.global_pointer = bert4torch.layers.EfficientGlobalPointer(
            hidden_size=self.bert.config.hidden_size,
            heads=num_tags, head_size=ner_hidden_size,
        )

    def forward(self, *args, **kwargs):
        output = self.bert(*args, **kwargs)
        logits = self.global_pointer(
            output.last_hidden_state, mask=kwargs['attention_mask'],
        )
        return logits


class MyLoss(bert4torch.losses.MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        # (batch_size * num_tags, seq_len * seq_len)
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)
        return super().forward(y_pred, y_true)


def compute_metrics(eval_pred: [transformers.EvalPrediction]) -> Dict:
    n_gold = 0
    n_pred = 0
    n_correct = 0

    for i, y_true in enumerate(eval_pred.label_ids):
        # preds = set(zip(*np.where(eval_pred.predictions[i] > 0)))
        # golds = set(zip(*np.where(y_true > 0)))
        # the transformation above is done in custom eval_loop
        preds = eval_pred.predictions[i]
        golds = y_true
        n_gold += len(golds)
        n_pred += len(preds)
        n_correct += len(golds & preds)

    p = n_correct / n_pred if n_pred else 0
    r = n_correct / n_gold if n_gold else 0
    f = 2 * n_correct / (n_pred + n_gold) if n_pred + n_gold else 0
    return {'precision': p, 'recall': r, 'f1': f}


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / 'data' / 'china-people-daily-ner-corpus'
MODEL_DIR = ROOT_DIR / 'pretrained_models' / 'bert-base-chinese'

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

label2id = {"LOC": 0, "ORG": 1, "PER": 2}
max_length = 256
collate_fn = make_collate_fn(
    tokenizer=tokenizer, max_length=max_length, label2id=label2id
)
train_dataset = MyDataset(str(DATA_DIR / 'example.train'))
eval_dataset = MyDataset(str(DATA_DIR / 'example.dev'))
test_dataset = MyDataset(str(DATA_DIR / 'example.test'))


def model_init():
    return Model(MODEL_DIR, num_tags=len(label2id))


MyTrainer = utils.make_simple_trainer(
    MyLoss(),
    additional_overwrites=[global_pointer_utils.GlobalPointerEval],
    use_simple_optimizer=True,
    use_simple_scheduler=True,
)

batch_size = 16
training_args = transformers.TrainingArguments(
    output_dir='ner_global_pointer_output',
    label_names=['labels'],

    num_train_epochs=20,
    per_device_train_batch_size=batch_size,
    learning_rate=2e-5,

    per_device_eval_batch_size=batch_size,
    evaluation_strategy=transformers.IntervalStrategy.EPOCH,
    eval_steps=500,
    # https://discuss.huggingface.co/t/cuda-out-of-memory-during-evaluation-but-training-is-fine/1783
    eval_accumulation_steps=100,

    save_strategy=transformers.IntervalStrategy.EPOCH,
    save_steps=500,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='f1',

    # disable_tqdm=True,
    report_to=[],
)
trainer = MyTrainer(
    model_init=model_init,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    args=training_args,
    compute_metrics=compute_metrics,
)
trainer.train()
for dataset, prefix in [
    (train_dataset, 'final_train'),
    (eval_dataset, 'final_eval'),
    (test_dataset, 'final_test'),
]:
    print(trainer.evaluate(dataset, metric_key_prefix=prefix))

"""
{'final_train_loss': 0.025439348071813583, 'final_train_precision': 0.9941428316584409, 'final_train_recall': 0.9933749257278669, 'final_train_f1': 0.9937587303474307, 'final_train_runtime': 253.579, 'final_train_samples_per_second': 82.278, 'final_train_steps_per_second': 5.142, 'epoch': 20.0}
{'final_eval_loss': 0.2312776893377304, 'final_eval_precision': 0.9690301548492257, 'final_eval_recall': 0.9552758435993572, 'final_eval_f1': 0.9621038435603506, 'final_eval_runtime': 25.5128, 'final_eval_samples_per_second': 90.856, 'final_eval_steps_per_second': 5.683, 'epoch': 20.0}
{'final_test_loss': 0.27641570568084717, 'final_test_precision': 0.9548104956268222, 'final_test_recall': 0.9477769008155749, 'final_test_f1': 0.9512806971217322, 'final_test_runtime': 54.7903, 'final_test_samples_per_second': 84.614, 'final_test_steps_per_second': 5.293, 'epoch': 20.0}
"""