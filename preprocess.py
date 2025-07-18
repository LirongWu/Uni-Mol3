import os
import csv
import lmdb
import time
import pickle
import random
import linecache

import re
import torch
import argparse
import numpy as np
from rdkit import Chem


def merge_lmdb(lmdb_path, split, task_type, num_parts=24):
    if split == 'train':
        data_path = lmdb_path + '/train.lmdb'
    elif split == 'test':
        data_path = lmdb_path + '/test_{}.lmdb'.format(task_type)

    try:
        os.remove(data_path)
    except:
        pass
    env_0 = lmdb.open(
        data_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_0 = env_0.begin(write = True)

    num = 0

    for i in range(num_parts):
        env = lmdb.open(
            lmdb_path + '/_data_{}/3d_smiles_reac/train_{}.lmdb'.format(split, i),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin()
        
        for idx in range(len(list(txn.cursor().iternext(values=False)))):
            datapoint_pickled = txn.get(f'{idx}'.encode("ascii"))
            data = pickle.loads(datapoint_pickled)
            tokens = data['molecular']

            if split == 'test':
                if task_type == 'forward':
                    output = ".".join(tokens['reagent_3d_smiles']) + '>' + ".".join(tokens['reactant_3d_smiles']) + '-->>>>' + ".".join([tokens['product_1d_smiles']])
                    txn_0.put(f'{num}'.encode("ascii"), pickle.dumps(output))
                    num += 1
                elif task_type == 'reverse':
                    output = ".".join([tokens['product_3d_smiles']]) + '-->>>>' + ".".join(tokens['reactant_1d_smiles'])
                    txn_0.put(f'{num}'.encode("ascii"), pickle.dumps(output))
                    num += 1
                elif task_type == 'condition' and tokens['reagent_1d_smiles'] is not None:
                    output = ".".join(tokens['reactant_3d_smiles']) + '>' + ".".join([tokens['product_3d_smiles']]) + '-->>>>' + ".".join(tokens['reagent_1d_smiles'])
                    txn_0.put(f'{num}'.encode("ascii"), pickle.dumps(output))
                    num += 1
            else:
                txn_0.put(f'{num}'.encode("ascii"), pickle.dumps(tokens))
                num += 1

    print(num, len(list(txn.cursor().iternext(values=False))))
    txn_0.commit()
    env_0.close()

def get_canonical_smile(smiles):
    if smiles == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_smi = Chem.MolToSmiles(mol)
    return canonical_smi

def remove_atom_mapping(smiles):
    return re.sub(r':\d+', '', smiles)

def simplify_smiles(smiles):
    smiles = remove_atom_mapping(smiles)
    mol = Chem.MolFromSmiles(smiles)

    if mol is None or count_atoms_from_smiles(smiles) > 150 or count_atoms_from_smiles(smiles) == 0:
        return None

    standard_smiles = Chem.MolToSmiles(mol, allHsExplicit=False, canonical=False)
    return standard_smiles

def count_atoms_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    atom_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            atom_count += 1
    return atom_count
    
def get_map(smi):
    mol = Chem.MolFromSmiles(smi)
    id_list = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    return id_list, mol

def get_list(atoms_map_reactant, atoms_map_product):
    atoms_map_product_dict = {atoms_map_product[i]: i for i in range(len(atoms_map_product))}
    id_list = [atoms_map_product_dict[i] for i in atoms_map_reactant if i!= 0 ]
    return id_list

def reaction_remap(reactant, product):
    product_map, product_mol = get_map(product)
    reactant_map, reactant_mol = get_map(reactant)

    list_product = get_list(reactant_map, product_map)
    nm = Chem.RenumberAtoms(product_mol, list_product)

    return Chem.MolToSmiles(reactant_mol, canonical=False), Chem.MolToSmiles(nm, canonical=False)

def convert_to_smiles(reaction_smiles):
    reactants_str, reagents_str, products_str = reaction_smiles.split(">")

    '''
    reactants_str = get_canonical_smile(reactants_str)
    if reactants_str is None or get_canonical_smile(products_str) is None:
        return None

    reactant_atom_num, _ = get_map(reactants_str)
    product_atom_num, _ = get_map(products_str)

    if len(products_str.split(".")) != 1:
        return None

    if 0 in product_atom_num:
        return None

    reactant_atom_num = [x for x in reactant_atom_num if x != 0]
    if len(set(reactant_atom_num)) != len(reactant_atom_num) or len(set(product_atom_num)) != len(product_atom_num):
        return None

    if any(element not in reactant_atom_num for element in product_atom_num):
        return None

    reactants_str, products_str = reaction_remap(reactants_str, products_str)
    '''

    reactants_list = reactants_str.split(".")
    reactants = []
    for reactant in reactants_list:
        reactant = simplify_smiles(reactant)
        if reactant:
            reactants.append(reactant)
        else:
            return None
    new_reactants_str = ".".join(reactants)

    reagents_list = reagents_str.split(".")
    reagents = []
    for reagent in reagents_list:
        reagent = simplify_smiles(reagent)
        if reagent:
            reagents.append(reagent)
        else:
            reagents = []
            break
    new_reagents_str = ".".join(reagents)

    new_products_str = simplify_smiles(products_str)
    if new_products_str is None or count_atoms_from_smiles(new_products_str) <= 1:
        return None

    if new_products_str in reactants:
        return None

    new_reaction = f"{new_reactants_str}>{new_reagents_str}>{new_products_str}"
    atom_reaction = f"{reactants_str}>{reagents_str}>{products_str}"
    return [new_reaction, atom_reaction]


def save_lmdb(file_path, lmdb_path, split, split_total=24, data_num=0):
    time_start = time.time()
    try:
        os.remove(lmdb_path + 'train_{}.lmdb'.format(split))
    except:
        pass
    env_new = lmdb.open(
        lmdb_path + 'train_{}.lmdb'.format(split),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write = True)

    mol_num = 0
    line_num = 0
    reac_num = 0
    spilt_num = data_num // split_total

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_num += 1

            if split == 0 and line_num > spilt_num:
                continue
            if split == (split_total - 1) and line_num <= spilt_num * split:
                continue
            if  split != 0 and split != (split_total - 1):
                if line_num <= split * spilt_num or line_num > (split + 1) * spilt_num:
                    continue

            # line_s = line.strip().split()[0]
            line_s = line
            # line_s = line_s.split(">")[0] + '>>' + line_s.split(">")[-1]
            try:
                line = convert_to_smiles(line_s)
            except Exception as e:
                line = None
                with open('error_log.txt', 'a+') as f:
                    f.write('{}_{}: {}\n'.format(split, line_num, str(e)))
                    f.flush()

            if line:
                reactants_str, reagents_str, product = line[0].split(">")
                
                reactants_list = reactants_str.split(".")
                reactant_long = []
                reactant_short = []
                for reactant in reactants_list:
                    atom_num = count_atoms_from_smiles(reactant)
                    if atom_num > 1:
                        reactant_long.append(reactant)
                    elif atom_num == 1:
                        reactant_short.append(reactant)
                if len(reactant_long) == 0:
                    if len(reactant_short) > 1:
                        txn_write.put(f'{mol_num}'.encode("ascii"), pickle.dumps(".".join(reactant_short)))
                        mol_num += 1
                else:
                    for i, reactant in enumerate(reactant_long):
                        if i == 0:
                            if len(reactant_short) != 0:
                                txn_write.put(f'{mol_num}'.encode("ascii"), pickle.dumps(reactant + '.' + ".".join(reactant_short)))
                                mol_num += 1   
                            else:
                                txn_write.put(f'{mol_num}'.encode("ascii"), pickle.dumps(reactant))
                                mol_num += 1                      
                        else:
                            txn_write.put(f'{mol_num}'.encode("ascii"), pickle.dumps(reactant))
                            mol_num += 1                    


                txn_write.put(f'{mol_num}'.encode("ascii"), pickle.dumps(product))
                mol_num += 1

                reagents_list = reagents_str.split(".")
                reagent_long = []
                reagent_short = []
                for reagent in reagents_list:
                    atom_num = count_atoms_from_smiles(reagent)
                    if atom_num > 1:
                        reagent_long.append(reagent)
                    elif atom_num == 1:
                        reagent_short.append(reagent)
                if len(reagent_long) == 0:
                    if len(reagent_short) > 1:
                        txn_write.put(f'{mol_num}'.encode("ascii"), pickle.dumps(".".join(reagent_short)))
                        mol_num += 1
                else:
                    for i, reagent in enumerate(reagent_long):
                        if i == 0:
                            if len(reagent_short) != 0:
                                txn_write.put(f'{mol_num}'.encode("ascii"), pickle.dumps(reagent + '.' + ".".join(reagent_short)))
                                mol_num += 1   
                            else:
                                txn_write.put(f'{mol_num}'.encode("ascii"), pickle.dumps(reagent))
                                mol_num += 1                      
                        else:
                            txn_write.put(f'{mol_num}'.encode("ascii"), pickle.dumps(reagent))
                            mol_num += 1                    

                reac_num += 1

            #     print('-------------------')
            #     print(line_num)
            #     print(line_s)
            #     print(line[0])
            #     print(line[1])
            #     print(reactants_list)
            #     print(reagents_list)
            #     print(product)

            # if (line_num) % 5000 == 0:
            #     break
            if (line_num - 1) % 100000 == 0:
                print(split, line_num, time.time() - time_start)


    print(line_num, reac_num, mol_num)
    print(line_s)
    print(line)
    print(reactants_list)
    print(reagents_list)
    print(product)

    txn_write.commit()
    env_new.close()


def load_lmdb(file_path, lmdb_path, split, split_total=24, data_num=0):
    time_start = time.time()
    try:
        os.remove(lmdb_path + '3d_smiles_reac/' + 'train_{}.lmdb'.format(split))
    except:
        pass
    env_new = lmdb.open(
        lmdb_path + '3d_smiles_reac/' + 'train_{}.lmdb'.format(split),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write = True)

    env = lmdb.open(
        lmdb_path + '3d_smiles/' + 'train_{}.lmdb'.format(split),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()

    mol_num = 0
    line_num = 0
    reac_num = 0
    spilt_num = data_num // split_total

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_num += 1

            if split == 0 and line_num > spilt_num:
                continue
            if split == (split_total - 1) and line_num <= spilt_num * split:
                continue
            if  split != 0 and split != (split_total - 1):
                if line_num <= split * spilt_num or line_num > (split + 1) * spilt_num:
                    continue

            # line_s = line.strip().split()[0]
            line_s = line
            # line_s = line_s.split(">")[0] + '>>' + line_s.split(">")[-1]
            try:
                line = convert_to_smiles(line_s)
            except Exception as e:
                line = None
                with open('error_log.txt', 'a+') as f:
                    f.write('{}_{}: {}\n'.format(split, line_num, str(e)))
                    f.flush()

            if line:
                reactants_str, reagents_str, product = line[0].split(">")
                reactant_list = []
                reactant_list_smiles = []
                reagent_list = []
                reagent_list_smiles = []
                
                reactants_list = reactants_str.split(".")
                reactant_long = []
                reactant_short = []
                for reactant in reactants_list:
                    atom_num = count_atoms_from_smiles(reactant)
                    if atom_num > 1:
                        reactant_long.append(reactant)
                    elif atom_num == 1:
                        reactant_short.append(reactant)
                if len(reactant_long) == 0:
                    if len(reactant_short) > 1:
                        reactant_list.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles_3d'])
                        reactant_list_smiles.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles'])
                        mol_num += 1
                else:
                    for i, reactant in enumerate(reactant_long):
                        if i == 0:
                            if len(reactant_short) != 0:
                                reactant_list.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles_3d'])
                                reactant_list_smiles.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles'])
                                mol_num += 1   
                            else:
                                reactant_list.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles_3d'])
                                reactant_list_smiles.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles'])
                                mol_num += 1                      
                        else:
                            reactant_list.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles_3d'])
                            reactant_list_smiles.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles'])
                            mol_num += 1                    

                product_3d_smiles = pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles_3d']
                product_1d_smiles = pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles']
                mol_num += 1

                reagents_list = reagents_str.split(".")
                reagent_long = []
                reagent_short = []
                for reagent in reagents_list:
                    atom_num = count_atoms_from_smiles(reagent)
                    if atom_num > 1:
                        reagent_long.append(reagent)
                    elif atom_num == 1:
                        reagent_short.append(reagent)
                if len(reagent_long) == 0:
                    if len(reagent_short) > 1:
                        reagent_list.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles_3d'])
                        reagent_list_smiles.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles'])
                        mol_num += 1
                else:
                    for i, reagent in enumerate(reagent_long):
                        if i == 0:
                            if len(reagent_short) != 0:
                                reagent_list.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles_3d'])
                                reagent_list_smiles.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles'])
                                mol_num += 1   
                            else:
                                reagent_list.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles_3d'])
                                reagent_list_smiles.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles'])
                                mol_num += 1                      
                        else:
                            reagent_list.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles_3d'])
                            reagent_list_smiles.append(pickle.loads(txn.get(f'{mol_num}'.encode("ascii")))['smiles'])
                            mol_num += 1                    

                if 'None' not in reagent_list and 'None' not in reactant_list:
                    reaction_str = ".".join(reagent_list) + '>' + ".".join(reactant_list) + '-->>>>' + product
                    molecular_dict = {
                        "reactant_3d_smiles": reactant_list,
                        "reactant_1d_smiles": reactant_list_smiles,
                        "reagent_3d_smiles": reagent_list,
                        "reagent_1d_smiles": line[1].split(">")[1].split("."),
                        "product_3d_smiles": product_3d_smiles,
                        "product_1d_smiles": product_1d_smiles,
                    }
                    save_out = {
                        "reaction_str": line_s,
                        "reaction_aligh": line[1],
                        "reaction_preprocess": line[0],
                        "molecular": molecular_dict,
                        "reaction_final": reaction_str
                    }
                    txn_write.put(f'{reac_num}'.encode("ascii"), pickle.dumps(save_out))
                    # print('-----------------')
                    # print(line_num)
                    # print(reactants_list)
                    # print(reactant_list)
                    # print(reagents_list)
                    # print(reagent_list)
                    # print(product)
                    # print(reaction_str)
                    reac_num += 1
                else:
                    with open('../error_log.txt', 'a+') as f:
                        f.write('{}\n'.format(line))
                        f.flush()


            # if (line_num) % 5000 == 0:
            #     break
            if (line_num - 1) % 100000 == 0:
                print(split, line_num, time.time() - time_start)


    print(line_num, reac_num, mol_num)
    print(line_s)
    print(line)
    print(reactants_list)
    print(reactant_list)
    print(reagents_list)
    print(reagent_list)
    print(product)

    txn_write.commit()
    env_new.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--task_type", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--data_num", type=int)
    args = parser.parse_args()

    folder_path = '{}/_data_{}/3d_smiles_reac/'.format(args.dataset, args.split)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if args.mode == 'save':
        save_lmdb('{}/{}.smi'.format(args.dataset, args.split), '{}/_data_{}/'.format(args.dataset, args.split), 0, 1, args.data_num)
    elif args.mode == 'load':
        load_lmdb('{}/{}.smi'.format(args.dataset, args.split), '{}/_data_{}/'.format(args.dataset, args.split), 0, 1, args.data_num)
    elif args.mode == 'merge':
        merge_lmdb(args.dataset, args.split, args.task_type, 1)