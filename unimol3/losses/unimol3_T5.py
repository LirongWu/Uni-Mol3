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


def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'].upper() == row['{}{}'.format(base, i)].upper():
            return i
    return 0

def calculate_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return 0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    return similarity

def calculate_bleu_score(smiles1, smiles2):
    sf = SmoothingFunction()
    score = sentence_bleu([list(smiles1)], list(smiles2), smoothing_function=sf.method1)
    return score
    
def get_mol_similarity(row, base, max_rank):
    mol_sim = 0
    for i in range(1, max_rank+1):
        sim = calculate_similarity(row['target'], row['{}{}'.format(base, i)])
        if sim > mol_sim:
            mol_sim = sim
    return mol_sim

def get_levenshtein_dis(row, base, max_rank):
    leve_dis = 10000
    for i in range(1, max_rank+1):
        dis = Levenshtein.distance(row['target'], row['{}{}'.format(base, i)])
        if dis < leve_dis:
            leve_dis = dis
    return leve_dis

def get_bleu_score(row, base, max_rank):
    bleu_score = 0
    for i in range(1, max_rank+1):
        score = calculate_bleu_score(row['target'], row['{}{}'.format(base, i)])
        if score > bleu_score:
            bleu_score = score
    return bleu_score
    
def standize(smiles):
    try:
        canon_smiles = Chem.CanonSmiles(smiles)
        return canon_smiles
    except:
        return ''


@register_loss("unimol3_T5")
class UniMol3T5_Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.mask_token_id = 4
        self.num_beams = task.args.num_beams
        self.num_preds = task.args.num_preds
        self.task_type = task.args.task_type

    def forward(self, model, sample, reduce=True):
        input_key = "batched_data"

        # used for lm head
        masked_tokens = sample[input_key]["labels"].ne(model.T5Model.config.pad_token_id)
        masked_cnt = masked_tokens.long().sum()

        # print(sample[input_key]["input_ids"].shape, sample[input_key]["attention_mask"].shape, sample[input_key]["labels"].shape, masked_tokens.shape)
        # print(sample[input_key]["input_ids"][0], sample[input_key]["attention_mask"][0], sample[input_key]["labels"][0], masked_tokens[0])

        # calculate masked token loss...
        net_output = model(**sample[input_key])
        masked_token_loss = net_output.loss
        loss = masked_token_loss

        if masked_tokens.any():
            target = sample[input_key]["labels"][masked_tokens]
            masked_pred = net_output.logits[masked_tokens].argmax(dim=-1)
            masked_hit = (masked_pred == target).long().sum().data
        else:
            masked_hit = 0

        logging_output = {
            "sample_size": 1,
            "bsz": sample[input_key]["labels"].size(0),
            "seq_len": sample[input_key]["labels"].size(1)
            * sample[input_key]["labels"].size(0),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit,
            "masked_token_cnt": masked_cnt,
            "loss": loss.data,
        }

        if not self.training and self.task_type != 'pretrain' and self.task_type != 'reaction':
            task_specific_params = {
                "early_stopping": True,
                "max_length": 512,
                "num_beams": self.num_beams,
                "num_return_sequences": self.num_preds,
                "decoder_start_token_id": model.T5Model.config.decoder_start_token_id,
            }

            targets = []
            for i in range(sample[input_key]["labels"].shape[0]):
                target = model.tokenizer.decode(sample[input_key]["labels"][i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                targets.append(standize(target.replace(" ","")))
                # print(i, target)

            del sample[input_key]['labels']
            outputs = model.T5Model.generate(**sample[input_key], **task_specific_params)
            predictions = [[] for i in range(self.num_preds)]
            for i, pred in enumerate(outputs):
                pred = model.tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                predictions[i % self.num_preds].append(pred.replace(" ",""))
                # print(i//self.num_preds, i % self.num_preds, pred.replace(" ",""))


            test_df = pd.DataFrame(targets, columns=['target'])
            for i, preds in enumerate(predictions):
                test_df['prediction_{}'.format(i + 1)] = preds
                test_df['prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(standize)
            test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'prediction_', self.num_preds), axis=1)
            test_df['mol_sim'] = test_df.apply(lambda row: get_mol_similarity(row, 'prediction_', self.num_preds), axis=1)
            test_df['leve_dis'] = test_df.apply(lambda row: get_levenshtein_dis(row, 'prediction_', self.num_preds), axis=1)
            test_df['bleu_score'] = test_df.apply(lambda row: get_bleu_score(row, 'prediction_', self.num_preds), axis=1)

            correct = 0
            invalid_smiles = 0
            for i in range(1, self.num_preds+1):
                correct += (test_df['rank'] == i).sum()
                invalid_smiles += (test_df['prediction_{}'.format(i)] == '').sum()
                logging_output["corrcet_{}".format(i)] = correct
                logging_output["invalid_smiles_{}".format(i)] = invalid_smiles

            logging_output["mol_sim"] = test_df['mol_sim'].sum()
            logging_output["mol_sim_nonzero"] = test_df[test_df['mol_sim'] != 0]['mol_sim'].sum()
            logging_output["mol_sim_num"] = (test_df['mol_sim'] != 0).sum()
            logging_output["leve_dis"] = test_df['leve_dis'].sum()
            logging_output["bleu_score"] = test_df['bleu_score'].sum()

        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=5)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "masked_token_loss", masked_loss / sample_size, sample_size, round=5
        )

        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / (sum(log.get("masked_token_cnt", 0) for log in logging_outputs) + 1e-6)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=5)

        if "valid" in split or "test" in split:
            try:
                num_preds = 5
                for i in range(1, num_preds+1):
                    correct = sum(log.get("corrcet_{}".format(i), 0) for log in logging_outputs)
                    invalid_smiles = sum(log.get("invalid_smiles_{}".format(i), 0) for log in logging_outputs)
                    # print('Top-{}: {:.1f}% || Invalid {:.2f}%'.format(i, correct/bsz*100, invalid_smiles/bsz/i*100))
                    metrics.log_scalar("Top{}-acc".format(i), correct/bsz*100, 1, round=5)
                    metrics.log_scalar("Top{}-invalid".format(i), invalid_smiles/bsz/i*100, sample_size, round=5)
            except:
                print("Skip Generation ...")
    
        mol_sim = sum(log.get("mol_sim", 0) for log in logging_outputs)
        mol_sim_nonzero = sum(log.get("mol_sim_nonzero", 0) for log in logging_outputs)
        mol_sim_num = sum(log.get("mol_sim_num", 0) for log in logging_outputs)
        metrics.log_scalar("sim_score", mol_sim / bsz, sample_size, round=5)
        if mol_sim_nonzero == 0:
            metrics.log_scalar("sim_score_nonzero", 0.0, sample_size, round=5)
        else:
            metrics.log_scalar("sim_score_nonzero", mol_sim_nonzero / mol_sim_num, sample_size, round=5)
        leve_dis = sum(log.get("leve_dis", 0) for log in logging_outputs)
        metrics.log_scalar("dis_score", leve_dis / bsz, sample_size, round=5)
        bleu_score = sum(log.get("bleu_score", 0) for log in logging_outputs)
        metrics.log_scalar("bleu_score", bleu_score / bsz, sample_size, round=5)
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
