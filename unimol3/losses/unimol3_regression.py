# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import pandas as pd
from rdkit import Chem

import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


import re
import torch
import argparse
import Levenshtein
from rdkit import DataStructs
from rdkit.Chem import AllChem
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from unimol3.models import UniMol3Model, base_architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@register_loss("unimol3_regression")
class UniMol3Regression_Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.task_type = task.args.task_type

    def forward(self, model, sample, reduce=True):
        input_key = "batched_data"
        net_output = model(**sample[input_key], regression=True)
        loss = net_output['loss']
        predictions = net_output['logits']

        logging_output = {
            "sample_size": 1,
            "bsz": sample[input_key]["labels"].size(0),
            "loss": loss.data,
        }

        if not self.training:
            targets = sample[input_key]['labels'].view(-1)
            predictions = predictions.view(-1)
            logging_output["MAE"] = torch.abs(targets - predictions).sum()
            logging_output["RMSE"] = ((targets - predictions) ** 2).sum()
            # Test1 Avg 36.05175914095773
            # Test2 Avg 34.16499289931558
            # Test3 Avg 33.43453210191305
            # Test4 Avg 30.796053244836656
            logging_output["SST"] = ((targets - 30.796053244836656) ** 2).sum()

        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=5)

        if "valid" in split or "test" in split:
            MAE = sum(log.get("MAE", 0) for log in logging_outputs)
            metrics.log_scalar("MAE", MAE / bsz, sample_size, round=5)
            RMSE = sum(log.get("RMSE", 0) for log in logging_outputs)
            metrics.log_scalar("RMSE", torch.sqrt(RMSE / bsz), sample_size, round=5)
            SST = sum(log.get("SST", 0) for log in logging_outputs)
            metrics.log_scalar("R2", 1 - RMSE/SST, sample_size, round=5)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
