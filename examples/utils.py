import pathlib
from typing import List, Optional, Type, Callable, Union, Dict

import torch
import transformers
import torch.nn.functional as F


class SimpleOptimizer(transformers.Trainer):
    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate,
        )


class SimpleScheduler(transformers.Trainer):
    def create_scheduler(self, num_training_steps, optimizer=None):
        # overwrites the default scheduler and does nothing instead
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda x: 1,
        )


def make_simple_trainer(
    loss_fct: Callable,
    additional_overwrites: Optional[List[Type[transformers.Trainer]]] = None,
    use_simple_optimizer: bool = True,
    use_simple_scheduler: bool = True,
) -> Type[transformers.Trainer]:
    """
    Huggingface Trainer has reasonable default optimizer and scheduler, which are
    probably more suitable for pretrained models. But anyway it is advisable to try
    the simplest method first.

    `additional_overwrites` is a list of subclasses of Trainer to inject custom behaviors.
    """
    mro = additional_overwrites if additional_overwrites else []

    if use_simple_optimizer:
        mro.append(SimpleOptimizer)
    if use_simple_scheduler:
        mro.append(SimpleScheduler)
    if not mro:
        mro = [transformers.Trainer]

    print('Trainer mro:', mro)

    class MyTrainer(*mro):
        def compute_loss(self, model, inputs, return_outputs=False):
            """`inputs` must be a dict with `labels` as the key for labels"""
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            loss = loss_fct(outputs, labels)
            return (loss, {'outputs': outputs}) if return_outputs else loss

    return MyTrainer


def load_item2id(fp) -> Dict[str, int]:
    with open(fp, encoding='utf8') as f:
        items = f.read().split('\n')
    items = [item for item in items if item]
    return {item: idx for idx, item in enumerate(items)}


def load_id2item(fp) -> Dict[int, str]:
    d = load_item2id(fp)
    return {v: k for k, v in d.items()}


def load_best_state_dict(dir):
    """Loads the only checkpoint from output_dir of Trainer
    TODO: check trainer_state.json
    """
    dir = pathlib.Path(dir)
    checkpoints = list(dir.glob('checkpoint-*'))
    assert len(checkpoints) == 1
    return torch.load(checkpoints[0] / 'pytorch_model.bin')


def get_multihot_encodings(
    labels: Union[List[int], List[List[int]]], num_classes: int = -1,
) -> torch.Tensor:
    """
    Examples
    --------
    >>> get_multihot_encodings([[1, 2, 4], [2, 3]])
    tensor([[0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0]])
    """
    if isinstance(labels[0], list):
        labels: List[List[int]]
        if num_classes == -1:
            num_classes = max(max(sub_list) for sub_list in labels) + 1
        return torch.cat(
            [
                get_multihot_encodings(sub_list, num_classes=num_classes).view(1, -1)
                for sub_list in labels
            ]
        )

    onehot_encodings = F.one_hot(torch.LongTensor(labels), num_classes=num_classes)
    multihot_encodings = onehot_encodings.sum(dim=0).float()
    return multihot_encodings
