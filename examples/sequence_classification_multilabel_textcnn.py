import os
import pathlib
import time
from typing import List, Tuple, Dict

import bert4torch.snippets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F

import utils


class MyDataset(bert4torch.snippets.ListDataset):
    @staticmethod
    def load_data(filename) -> List[Tuple]:
        """
        file format like fasttext without tokenization

        __label__xx __label__oo sentence0
        __label__oo  sentence1
        """
        data = []
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                labels = []
                while True:
                    if line.startswith('__label__'):
                        label, line = line.split(' ', maxsplit=1)
                        labels.append(label[len('__label__'):])
                    else:
                        assert '__label__' not in line
                        break
                data.append((line, labels))
        return data


def make_collate_fn(tokenizer, max_length, label2id):
    def collate_fn(data):
        texts, batch_labels = list(zip(*data))
        batch_encoding = tokenizer(
            texts, max_length=max_length, truncation=True, padding='max_length'
        )

        final_dict = {'inputs': batch_encoding.data['input_ids']}
        # final_dict = batch_encoding.data
        for k, v in final_dict.items():
            final_dict[k] = torch.LongTensor(v)

        batch_labels: List[List[int]] = [
            [label2id[label] for label in labels] for labels in batch_labels
        ]
        final_dict['labels'] = utils.get_multihot_encodings(
            batch_labels, num_classes=len(label2id)
        )
        return final_dict
    return collate_fn


def init_embedding(
    num_embeddings=None,
    embedding_dim=None,
    padding_idx=None,
    freeze=False,
):
    return nn.Embedding(
        num_embeddings, embedding_dim, padding_idx=padding_idx,
    )


class TextCNN(nn.Module):
    def __init__(
        self, num_classes: int,
        num_filters: int = 256,
        region_sizes: Tuple[int] = (2, 3, 4),
        classifier_dropout: float = 0.5,
        **embedding_params,
    ):
        super().__init__()
        self.embedding = init_embedding(**embedding_params)
        self.conv2ds = nn.ModuleList(
            nn.Conv2d(
                1, num_filters,
                kernel_size=(region_size, self.embedding.embedding_dim),
            )
            for region_size in region_sizes
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.linear = nn.Linear(num_filters * len(region_sizes), num_classes)

    @staticmethod
    def _conv_and_pool(x, conv2d):
        """(batch_size, 1, seq_len, embedding_dim)"""
        x = F.relu(conv2d(x)).squeeze(-1)  # (batch_size, num_filters, H_out)
        # explicit int(...) conversion is necessary for onnx
        x = F.max_pool1d(x, kernel_size=int(x.size(-1))).squeeze(-1)
        return x  # (batch_size, num_filters)

    def forward(self, inputs):
        """(batch_size, seq_len)"""
        x = self.embedding(inputs).unsqueeze(-3)
        x = torch.cat(
            [self._conv_and_pool(x, conv2d) for conv2d in self.conv2ds], dim=-1,
        )  # (batch_size, num_filters * len(region_sizes))
        x = self.dropout(x)
        x = self.linear(x)
        return x  # (batch_size, num_classes)


def compute_metrics(eval_pred: [transformers.EvalPrediction]) -> Dict:
    """micro"""
    preds = np.where(eval_pred.predictions > 0, 1, 0)
    n_correct = np.logical_and(
        preds == eval_pred.label_ids,
        eval_pred.label_ids == 1,
    ).sum()
    n_gold = eval_pred.label_ids.sum()
    n_pred = preds.sum()
    p = n_correct / n_pred if n_pred else 0
    r = n_correct / n_gold if n_gold else 0
    f = 2 * n_correct / (n_pred + n_gold) if n_pred + n_gold else 0
    return {'precision': p, 'recall': r, 'f1': f}


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

ROOT_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / 'data' / 'processed' / 'XXX'  # use your dataset
MODEL_DIR = ROOT_DIR / 'pretrained_models' / 'bert-base-chinese'

# use any tokenizer you like
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

label2id = utils.load_item2id(DATA_DIR / 'labels.txt')
max_length = 80
collate_fn = make_collate_fn(tokenizer=tokenizer, max_length=max_length, label2id=label2id)
train_dataset = MyDataset(str(DATA_DIR / 'train.txt'))
eval_dataset = MyDataset(str(DATA_DIR / 'eval.txt'))
test_dataset = MyDataset(str(DATA_DIR / 'test.txt'))
output_dir = 'textcnn'


def model_init():
    model = TextCNN(
        num_classes=len(label2id),
        num_embeddings=21128, embedding_dim=100, padding_idx=0,
    )
    model.load_state_dict(utils.load_best_state_dict(output_dir))
    return model


MyTrainer = utils.make_simple_trainer(
    nn.BCEWithLogitsLoss(),
    use_simple_optimizer=False,
    use_simple_scheduler=False,
)

batch_size = 96
training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    label_names=['labels'],

    num_train_epochs=100,
    per_device_train_batch_size=batch_size,

    learning_rate=1e-3,
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
