"""
Compared with https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch/blob/master/models/bert.py
"""
import os
import pathlib
from typing import List, Tuple, Dict

import bert4torch.snippets
import numpy as np
import torch
import torch.nn as nn
import transformers

import utils


class MyDataset(bert4torch.snippets.ListDataset):
    @staticmethod
    def load_data(filename) -> List[Tuple]:
        data = []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text, label = line.rsplit('\t', maxsplit=1)
                data.append((text, int(label)))
        return data


def make_collate_fn(tokenizer, max_length):
    def collate_fn(data):
        texts, labels = list(zip(*data))
        batch_encoding = tokenizer(
            texts, max_length=max_length, truncation=True, padding='max_length'
        )
        final_dict = batch_encoding.data
        final_dict['labels'] = np.array(labels)
        for k, v in final_dict.items():
            final_dict[k] = torch.LongTensor(v)
        return final_dict
    return collate_fn


class Model(nn.Module):
    def __init__(self, bert_path, num_labels=2, classifier_dropout=None):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(
            classifier_dropout if classifier_dropout is not None
            else self.bert.config.hidden_dropout_prob  # 0.1
        )
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, *args, **kwargs):
        output = self.bert(*args, **kwargs)
        pooled_output = output.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def compute_metrics(eval_pred: [transformers.EvalPrediction]) -> Dict:
    preds = np.argmax(eval_pred.predictions, axis=-1)
    acc = (preds == eval_pred.label_ids).sum() / len(preds)
    return {'accuracy': acc}


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ROOT_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / 'data' / 'THUCNews_sample'
MODEL_DIR = ROOT_DIR / 'pretrained_models' / 'bert-base-chinese'

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

max_length = 32
collate_fn = make_collate_fn(tokenizer=tokenizer, max_length=max_length)
train_dataset = MyDataset(str(DATA_DIR / 'train.txt'))
eval_dataset = MyDataset(str(DATA_DIR / 'dev.txt'))
test_dataset = MyDataset(str(DATA_DIR / 'test.txt'))


def model_init():
    return Model(MODEL_DIR, num_labels=10)


MyTrainer = utils.make_simple_trainer(
    nn.CrossEntropyLoss(),
    use_simple_optimizer=False,
    use_simple_scheduler=False,
)

batch_size = 128
training_args = transformers.TrainingArguments(
    output_dir='sequence_classification_output',
    label_names=['labels'],

    num_train_epochs=3,
    per_device_train_batch_size=batch_size,

    learning_rate=5e-5,
    weight_decay=0.01,
    adam_epsilon=1e-6,
    warmup_ratio=0.05,

    per_device_eval_batch_size=batch_size,
    evaluation_strategy=transformers.IntervalStrategy.STEPS,
    eval_steps=100,

    save_strategy=transformers.IntervalStrategy.STEPS,
    save_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,

    report_to=[],
)
trainer = MyTrainer(
    model_init=model_init,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    args=training_args,
    compute_metrics=compute_metrics,
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=10)]
)
trainer.train()
for dataset, prefix in [
    (train_dataset, 'final_train'),
    (eval_dataset, 'final_eval'),
    (test_dataset, 'final_test'),
]:
    print(trainer.evaluate(dataset, metric_key_prefix=prefix))

"""
{'final_train_loss': 0.06888076663017273, 'final_train_accuracy': 0.9784166666666667, 'final_train_runtime': 61.4491, 'final_train_samples_per_second': 2929.256, 'final_train_steps_per_second': 22.897, 'epoch': 2.7}
{'final_eval_loss': 0.19077110290527344, 'final_eval_accuracy': 0.94, 'final_eval_runtime': 3.5607, 'final_eval_samples_per_second': 2808.432, 'final_eval_steps_per_second': 22.187, 'epoch': 2.7}
{'final_test_loss': 0.1730271428823471, 'final_test_accuracy': 0.9466, 'final_test_runtime': 3.4297, 'final_test_samples_per_second': 2915.676, 'final_test_steps_per_second': 23.034, 'epoch': 2.7}
"""
