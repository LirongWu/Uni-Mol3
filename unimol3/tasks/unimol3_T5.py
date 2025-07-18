# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch 

import numpy as np
from unicore.data import (
    Dictionary,
    LMDBDataset,
    NestedDictionaryDataset,
    EpochShuffleDataset,
)
from unimol3.data import (
    T5ChemTasks,
    TaskPrefixDataset, 
    LineByLineTextDataset,
)
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("unimol3_T5")
class UniMol3_T5Task(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob-ratio",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--num-beams",
            default=10,
            type=int,
            help="Number of beams for beam search.",
        )
        parser.add_argument(
            "--weight-dir-path",
            type=str,
            default='None',
        )
        parser.add_argument(
            "--num-preds",
            default=5,
            type=int,
            help="The number of independently computed returned sequences for each element in the batch.",
        )
        parser.add_argument(
            "--task-type",
            type=str,
            default="pretrain",
            help="Task type to use. ('product', 'reactants', 'reagents', \
                'regression', 'classification', 'pretrain', 'mixed')",
        )

    def __init__(self, args, dictionary=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.args = args

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args, dictionary=None)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        task = T5ChemTasks[self.args.task_type]
        split_path = os.path.join(self.args.data, split + '.lmdb')  
        vocab_path = os.path.join(self.args.data, "vocab.pt")

        if self.args.task_type == 'reaction':
            self.args.mask_prob_ratio = 0.0
            
        if self.args.task_type == 'pretrain' or self.args.task_type == 'reaction':
            dataset = LMDBDataset(split_path)
            dataset = LineByLineTextDataset(
                dataset=dataset, 
                vocab_path=vocab_path,
                prefix=task.prefix,
                max_length=task.max_source_length,
                mask_ratio=self.args.mask_prob_ratio,
            )
        else:
            if self.args.task_type != 'mixed':
                prefix = task.prefix
            elif split =='test_forward':
                prefix = 'Product'
            elif split =='test_reverse':
                prefix = 'Reactants'
            elif split =='test_condition':
                prefix = 'Reagents'
            else:
                prefix = 'Mixed'
            dataset = LMDBDataset(split_path)
            dataset = TaskPrefixDataset(
                dataset=dataset,
                vocab_path=vocab_path,
                prefix=prefix,
                task_type = self.args.task_type,
                max_source_length=task.max_source_length,
                max_target_length=task.max_target_length,
            )
        dataset = NestedDictionaryDataset({"batched_data": dataset})
        
        if split == "train":
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)

        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        import shutil
        import random
        code_save_dir = args.save_dir+"/code/{}".format(random.randint(1000000000, 9999999999))
        os.makedirs(code_save_dir, exist_ok=True)
        shutil.copytree("./unimol3", code_save_dir+"/unimol3")

        model = models.build_model(args, self)
        if args.weight_dir_path != 'None':
            state_dict = torch.load(args.weight_dir_path)
            model.load_state_dict(state_dict['model'])
        if args.task_type == 'yield':
            model.register_regression_head()
        return model