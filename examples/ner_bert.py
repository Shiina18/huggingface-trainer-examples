"""
Adapted from transformers.BertForTokenClassification
"""
import os
import pathlib
from typing import List, Tuple, Dict, Set

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

            labels = np.ones(max_length) * label2id['O']
            for first, last, label in pos_labels[i]:
                if first in first_mapping and last in last_mapping:
                    labels[first_mapping[first]] = label2id['B-' + label]
                    labels[
                        first_mapping[first] + 1: last_mapping[last] + 1
                    ] = label2id['I-' + label]
            batch_labels.append(labels)

        final_dict = batch_encoding.data
        final_dict['labels'] = np.array(batch_labels)
        for k, v in final_dict.items():
            final_dict[k] = torch.LongTensor(v)

        return final_dict

    return collate_fn


class Model(nn.Module):
    def __init__(self, bert_path, num_labels, classifier_dropout=None):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(
            classifier_dropout if classifier_dropout is not None
            else self.bert.config.hidden_dropout_prob  # 0.1
        )
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, *args, **kwargs):
        output = self.bert(*args, **kwargs)
        sequence_output = output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class MyLoss(nn.CrossEntropyLoss):
    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1, self.num_labels)
        y_true = y_true.view(-1)
        return super().forward(y_pred, y_true)


def labels2entities(labels: List[str]) -> Set[Tuple]:
    """Input ['B-XX', 'I-XX', 'O', 'B-YY'], Output ('XX', 0, 2), ('YY', 3, 4)"""
    entity_type = None
    start = None
    entities = []

    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if entity_type is not None:
                entities.append((entity_type, start, i + 1))
            entity_type = label[len('B-'):]
            start = i
        elif (
            entity_type is not None
            and (label == 'O' or label[len('I-'):] != entity_type)
        ):
            entities.append((entity_type, start, i + 1))
            entity_type = None

    if entity_type is not None:
        entities.append((entity_type, start, i + 1))

    return set(entities)


def make_compute_metrics(id2label):
    def compute_metrics(eval_pred: [transformers.EvalPrediction]) -> Dict:
        """entity level"""
        n_gold = 0
        n_pred = 0
        n_correct = 0

        batch_preds = np.argmax(eval_pred.predictions, axis=-1)

        for i, y_true in enumerate(eval_pred.label_ids):
            preds = labels2entities([id2label[i] for i in batch_preds[i]])
            golds = labels2entities([id2label[i] for i in y_true])
            n_gold += len(golds)
            n_pred += len(preds)
            n_correct += len(golds & preds)

        p = n_correct / n_pred if n_pred else 0
        r = n_correct / n_gold if n_gold else 0
        f = 2 * n_correct / (n_pred + n_gold) if n_pred + n_gold else 0
        return {'precision': p, 'recall': r, 'f1': f}
    return compute_metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT_DIR = pathlib.Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / 'data' / 'china-people-daily-ner-corpus'
MODEL_DIR = ROOT_DIR / 'pretrained_models' / 'bert-base-chinese'

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

entity_types = ['LOC', 'ORG', 'PER']
labels = ['O'] + ['B-' + i for i in entity_types] + ['I-' + i for i in entity_types]
label2id = {x: i for i, x in enumerate(labels)}
id2label = {i: x for i, x in enumerate(labels)}
max_length = 256
collate_fn = make_collate_fn(
    tokenizer=tokenizer, max_length=max_length, label2id=label2id
)
train_dataset = MyDataset(str(DATA_DIR / 'example.train'))
eval_dataset = MyDataset(str(DATA_DIR / 'example.dev'))
test_dataset = MyDataset(str(DATA_DIR / 'example.test'))


def model_init():
    return Model(MODEL_DIR, num_labels=len(label2id))


MyTrainer = utils.make_simple_trainer(
    MyLoss(num_labels=len(label2id)),
    use_simple_optimizer=True,
    use_simple_scheduler=True,
)

batch_size = 16
training_args = transformers.TrainingArguments(
    output_dir='ner_bert_output',
    label_names=['labels'],

    num_train_epochs=20,
    per_device_train_batch_size=batch_size,
    learning_rate=2e-5,

    per_device_eval_batch_size=batch_size,
    evaluation_strategy=transformers.IntervalStrategy.EPOCH,
    eval_steps=5,
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
    compute_metrics=make_compute_metrics(id2label),
)
trainer.train()
print(trainer.evaluate(train_dataset, metric_key_prefix='final_train'))
print(trainer.evaluate(eval_dataset, metric_key_prefix='final_eval'))
print(trainer.evaluate(test_dataset, metric_key_prefix='final_test'))

"""
"""