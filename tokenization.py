# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import lmdb
import pickle
import argparse

import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

import torch 
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from unimol3.data import (
    KeyDataset,
    LMDBDataset,
    ConformerSampleDataset,
    RemoveHydrogenDataset,
    NormalizeDataset,
    CroppingDataset,
    Add2DConformerDataset,
    Unimol2FeatureDataset,
    IndexAtomDataset
)
from unimol3.models import UniMol3Model, base_architecture
from unicore.models import register_model_architecture
# from unicore.data import LMDBDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_args(parser):
    parser.add_argument(
        "--data-path",
        type=str,
        default='./data/USPTO-50k/_data_test/',
        # default='./data/ligands/',
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        default='./model/Tokenizer/checkpoint.pt',
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default='3d_smiles',
    )
    parser.add_argument(
        "--split",
        type=str,
        default='train',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=512,
        help="selected maximum number of atoms in a molecule",
    )
    parser.add_argument(
        "--data-mode",
        type=int,
        default=0,
    )

def map_atom_index_to_smiles_position(mol, smiles):
    atom_index_to_smiles_pos = {}
    start_pos = 0
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        try:
            smiles_idx = smiles[start_pos:].lower().find(atom_symbol.lower())
            atom_index_to_smiles_pos[atom_idx] = (smiles_idx + start_pos, atom_symbol)
            start_pos += smiles_idx + len(atom_symbol)
        except ValueError:
            print(f"Could not find atom {atom_symbol} in the parsed SMILES.")
    return atom_index_to_smiles_pos

def generate_3d_smiles(tokens, smiles):

    mol = Chem.MolFromSmiles(smiles)
    atom_index_to_smiles_pos = map_atom_index_to_smiles_position(mol, smiles)

    exist_num = 0
    for index in range(len(tokens)):
        start_pos = atom_index_to_smiles_pos[index][0] + exist_num
        atom_length = len(atom_index_to_smiles_pos[index][1])
        smiles = smiles[:start_pos] + '</{:04d}]'.format(int(tokens[index])) + smiles[start_pos+atom_length:]
        exist_num += len('</{:04d}]'.format(int(tokens[index]))) - atom_length

    return smiles

def reverse_tokenization(model, sample, smiles):
    out = model.classification_heads['quantizer'].indices_to_codes(sample.unsqueeze(0))
    out = model.classification_heads['decoder'](out)
    logits = model.lm_head(out)[0].argmax(dim=-1)

    pattern = r"</\d{4}]"
    atom_num = 0
    exist_num = 0
    for m in re.finditer(pattern, smiles):
        atom_index = int(logits.cpu().numpy()[atom_num])
        atom = AllChem.GetPeriodicTable().GetElementSymbol(atom_index)
        smiles = smiles[:m.start()-exist_num] + atom + smiles[m.end()-exist_num:]
        atom_num += 1
        exist_num += len('</0000]') - len(atom)

    return logits, smiles


# Tokenizing molecules (train/valid), ligands (vliad), reactants and products.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    raw_dataset = LMDBDataset(args.data_path, args.split)
    # raw_dataset = LMDBDataset(args.data_path + "test.lmdb")
    raw_dataset = Add2DConformerDataset(
        raw_dataset, "smi", "atoms", "coordinates", args.data_mode
    )
    smi_dataset = KeyDataset(raw_dataset, "smi")
    # sample conformer
    dataset = ConformerSampleDataset(
        raw_dataset, args.seed, "atoms", "coordinates", "coordinates_2d"
    )
    # remove H
    dataset = RemoveHydrogenDataset(
        dataset,
        "atoms",
        "coordinates",
        "coordinates_2d",
        True,
    )
    # cropping atom to max_atoms...
    dataset = CroppingDataset(
        dataset, args.seed, "atoms", "coordinates", "coordinates_2d", args.max_atoms
    )
    dataset = NormalizeDataset(dataset, "coordinates", "coordinates_2d", normalize_coord=True)
    # dataset -> coordinates, coordinates_2d

    token_dataset = KeyDataset(dataset, "atoms")
    origin_token_dataset = IndexAtomDataset(
        smi_dataset, token_dataset,
    )
    coord_dataset = KeyDataset(dataset, "coordinates")
    coordinates_2d = KeyDataset(dataset, "coordinates_2d")

    dataset = Unimol2FeatureDataset(
        smi_dataset=smi_dataset,
        token_dataset=origin_token_dataset,
        src_pos_dataset=coord_dataset,
        src_2d_pos_dataset=coordinates_2d,
        pad_idx=0,
        mask_idx=127,

        mask_token_prob=0.0,
        mask_pos_prob=0.0,
        noise=0.0,
        drop_feat_prob=0.0,
    )

    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        collate_fn=dataset.collater,
    )
    smi_loader = DataLoader(
        smi_dataset, 
        batch_size=args.batch_size,
    )

    args.mode = 'infer'
    args.trainmode = 0
    args.droppath_prob = 0.0
    args.masked_token_loss = 1.0
    
    base_architecture(args)
    model = UniMol3Model(args)
    model.register_tokenization_head()

    state_dict = torch.load(args.weight_path)
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval()

    folder_path = args.data_path + '{}/'.format(args.save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if args.data_mode == 1:
        outfilename = args.data_path + '{}/{}_reactant.lmdb'.format(args.save_path, args.split)
    elif args.data_mode == 0:
        outfilename = args.data_path + '{}/{}.lmdb'.format(args.save_path, args.split)
    else:
        outfilename = args.data_path + '{}/{}_target.lmdb'.format(args.save_path, args.split)
    try:
        os.remove(outfilename)
    except:
        pass
    env_new = lmdb.open(
        outfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write = True)
    num = 0
    batch_num = 0

    for (sample_batch, smiles_batch) in zip(tqdm(data_loader, desc="data"), tqdm(smi_loader, desc="smiles")):
        batch_num += 1
        try:
            sample_batch = {k: v.to(device) for k, v in sample_batch.items() if k != 'smiles'}
            logits_encoder, indices = model(sample_batch, tokenization=True)

            masked_tokens = sample_batch["src_token"].ne(0)
            target = sample_batch["src_token"][masked_tokens]

            masked_pred = logits_encoder[masked_tokens].argmax(dim=-1)
            masked_hit = (masked_pred == target).long().sum()
            masked_cnt = masked_tokens.long().sum()
            acc = masked_hit / masked_cnt

            # print(masked_hit, masked_cnt, acc)


            masked_hit = 0
            masked_cnt = 0

            for i in range(sample_batch['src_token'].size(0)):
                tokens = indices[i][masked_tokens[i]]
                smiles = smiles_batch[i]
                smiles_3d = generate_3d_smiles(tokens.cpu().numpy().tolist(), smiles)
                save_data = {'smiles': smiles, 'smiles_3d': smiles_3d}
                txn_write.put(f'{num}'.encode("ascii"), pickle.dumps(save_data))
                num += 1

                # print(smiles)
                # print(smiles_3d)

                pattern = r"</\d{4}]"
                matches = [(m.start(), m.end()) for m in re.finditer(pattern, smiles_3d)]
                sample = torch.tensor([int(smiles_3d[m[0]+2:m[1]-1]) for m in matches]).to(device)
                # print(sample)

                masked_pred, smiles = reverse_tokenization(model, sample, smiles_3d)
                # print(smiles)

                target = sample_batch["src_token"][i][masked_tokens[i]]
                masked_hit += (masked_pred == target).long().sum()
                masked_cnt += masked_tokens[i].long().sum()

            # acc = masked_hit / masked_cnt
            # print(masked_hit, masked_cnt, acc)
        except Exception as e:
            save_data = {'smiles': 'None', 'smiles_3d': 'None'}
            for i in range(args.batch_size):
                txn_write.put(f'{num+i}'.encode("ascii"), pickle.dumps(save_data))
            num += args.batch_size
            with open('log_file.txt', 'a+') as f:
                f.write('{}_{}: {}\n'.format(args.split, batch_num-1, str(e)))
                f.flush()

    print('process {} lines'.format(num))
    with open('log_file.txt', 'a+') as f:
        f.write(args.split + ': Start!!!\n')
        f.flush()

        txn_write.commit()
        env_new.close()

        f.write(args.split + ': Done!!!\n')
        f.flush()