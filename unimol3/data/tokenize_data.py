# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import random
from functools import lru_cache
from collections import Counter
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, NamedTuple

import torch
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from unicore.data import BaseWrapperDataset


pattern: str = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex: re.Pattern = re.compile(pattern)
TASK_PREFIX: List[str] = ['Yield:', 'Product:', 'Fill-Mask:', 'Classification:', 'Reagents:', 'Reactants:']

class TaskSettings(NamedTuple):
    prefix: str
    max_source_length: int
    max_target_length: int
    output_layer: str


T5ChemTasks: Dict[str, TaskSettings] = {
    'forward-sep': TaskSettings('Product', 400, 200, 'seq2seq'),
    'forward-mixed': TaskSettings('Product', 400, 200, 'seq2seq'),
    'reverse': TaskSettings('Reactants', 200, 300, 'seq2seq'),
    'condition': TaskSettings('Reagents', 400, 200, 'seq2seq'),
    'pretrain': TaskSettings('Fill-Mask:', 400, 200, 'seq2seq'),
    'reaction': TaskSettings('Fill-Mask:', 600, 600, 'seq2seq'),
    'yield': TaskSettings('Yield:', 500, 1, 'regression'),
    'mixed': TaskSettings('', 400, 300, 'seq2seq'),
}

class MolTokenizer(ABC, PreTrainedTokenizer):
    r"""
    An abstract class for all tokenizers. Other tokenizer should
    inherit this class
    """
    def __init__(
        self,
        vocab_file: Optional[str]=None,
        source_files: Optional[Union[str, List[str]]]=None,
        unk_token: str='<unk>',
        bos_token: str='<s>',
        pad_token: str="<pad>",
        eos_token: str='</s>',
        mask_token: str='<mask>',
        max_size: int=1200,
        task_prefixs: List[str]=[],
        **kwargs
    ) -> None:
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            pad_token=pad_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs)

        task_prefixs = TASK_PREFIX+task_prefixs
        self.create_vocab(
            vocab_file=vocab_file, 
            source_files=source_files, 
            vocab_size=max_size-len(task_prefixs)
            )
        if self.vocab:
            extra_to_add: int = max_size - len(self.vocab)
            cur_added_len: int = len(task_prefixs) + 9 # placeholder for smiles tokens
            for i in range(cur_added_len, extra_to_add):
                task_prefixs.append('<extra_task_{}>'.format(str(i)))
            self.add_tokens(['<extra_token_'+str(i)+'>' for i in range(9)]+task_prefixs+['>'], special_tokens=True)
            self.unique_no_split_tokens = sorted(
                set(self.unique_no_split_tokens).union(set(self.all_special_tokens))
            )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def merge_vocabs(
        self, 
        vocabs: List[Vocab], 
        vocab_size: Optional[int]=None,
    ) -> Vocab:
        """
        Merge individual vocabularies (assumed to be generated from disjoint
        documents) into a larger vocabulary.
        Args:
            vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
            vocab_size: `int` the final vocabulary size. `None` for no limit.
        Return:
            `torchtext.vocab.Vocab`
        """
        merged: Counter = sum([vocab.freqs for vocab in vocabs], Counter())
        special_tokens: List[str] = list(self.special_tokens_map.values())  # type: ignore
        for i in range(1024):
            special_tokens.append('</{:04d}]'.format(i))
        return Vocab(merged,
                    specials=special_tokens,
                    max_size=vocab_size-len(special_tokens) if vocab_size else vocab_size)

    def create_vocab(
        self, 
        vocab_file: Optional[str]=None,
        source_files: Optional[Union[str, List[str]]]=None,
        vocab_size: Optional[int]=None,
        ) -> None:
        """
        Create a vocabulary from current vocabulary file or from source file(s).
        Args:
            vocab_file (:obj:`string`, `optional`, defaults to ''):
                File containing the vocabulary (torchtext.vocab.Vocab class).
            source_files (:obj:`string`, `optional`, defaults to ''):
                File containing source data files, vocabulary would be built based on the source file(s).
            vocab_size: (:obj:`int`, `optional`, defaults to `None`):
                The final vocabulary size. `None` for no limit.
        """
        if vocab_file:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    "Can't find a vocabulary file at path '{}'.".format(vocab_file)
                )
            else:
                self.vocab: Vocab = self.merge_vocabs([torch.load(vocab_file)], vocab_size=vocab_size)

        elif source_files:
            if isinstance(source_files, str):
                if not os.path.isfile(source_files):
                    raise ValueError(
                        "Can't find a source file at path '{}'.".format(source_files)
                    )
                else:
                    source_files = [source_files]
            counter: Dict[int, Counter] = {}
            vocabs: Dict[int, Vocab] = {}
            for i, source_file in enumerate(source_files):
                counter[i] = Counter()
                with open(source_file) as rf:
                    for line in tqdm(rf, desc='Generating {}'.format(source_file)):
                        try:
                            items: List[str] = self._tokenize(line.strip())
                            counter[i].update(items)
                        except AssertionError:
                            print(line.strip())
                specials: List[str] = list(self.special_tokens_map.values()) # type: ignore
                vocabs[i] = Vocab(counter[i], specials=specials)
            self.vocab = self.merge_vocabs([vocabs[i] for i in range(len(source_files))], vocab_size=vocab_size)
        else:
            self.vocab = None

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    @abstractmethod
    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        """
        Tokenize a molecule or reaction
        """
        pass

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str) in an id using the vocab. """
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.stoi[token]

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        assert isinstance(self.vocab, Vocab),\
            'No vocabulary found! Need to be generated at initialization or using .create_vocab method.'
        return self.vocab.itos[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        out_string: str = "".join(tokens).strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A Mol sequence has the following format:
        - single sequence: ``<s> X </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def save_vocabulary(self, vocab_path: str) -> None:    # type: ignore
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        torch.save(self.vocab, vocab_path)


def custom_split(input_string):
    pattern = re.compile(r'(</\d{4}])|(.)')
    result = []
    for match in pattern.finditer(input_string):
        if match.group(1):
            result.append(match.group(1))
        else:
            result.append(match.group(2))
    # if len(result) == 0:
    #     result = list(input_string)
    return result


class SimpleTokenizer(MolTokenizer):
    r"""
    Constructs a simple, character-level tokenizer. Based on SMILES.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary (torchtext.vocab.Vocab class).
        source_files (:obj:`string`, `optional`, defaults to ''):
            File containing source data files, vocabulary would be built based on the source file(s).
        unk_token (:obj:`string`, `optional`, defaults to '<unk>'):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to '<s>'):
            string: a beginning of sentence token.
        pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`string`, `optional`, defaults to '</s>'):
            string: an end of sentence token.
        max_size: (:obj:`int`, `optional`, defaults to 100):
            The final vocabulary size. `None` for no limit.
        **kwargs：
            Arguments passed to `~transformers.PreTrainedTokenizer`
    """
    def __init__(self, vocab_file, max_size=100, **kwargs) -> None:
        super().__init__(vocab_file=vocab_file, max_size=max_size, **kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]: 
        return custom_split(text)


class LineByLineTextDataset(BaseWrapperDataset):
    def __init__(self, dataset, vocab_path, prefix, max_length, mask_ratio=0.15):
        self.dataset = dataset
        self.prefix = prefix
        self.max_length = max_length
        self.mask_mol_ratio = 0.15
        self.tokenizer = SimpleTokenizer(vocab_file=vocab_path)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mask_ratio)
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        if 'smiles_3d' in self.dataset[index].keys():
            source_str = self.dataset[index]['smiles_3d']
            target_str = self.dataset[index]['smiles']

            source_sample = self.tokenizer(
                            self.prefix+source_str,
                            max_length=self.max_length,
                            padding="do_not_pad",
                            truncation=True,
                            return_tensors='pt',
                        )

            target_sample = self.tokenizer(
                            target_str,
                            max_length=self.max_length,
                            padding="do_not_pad",
                            truncation=True,
                            return_tensors='pt',
                        )

            return [source_sample['input_ids'].squeeze(0), target_sample['input_ids'].squeeze(0)]
        
        elif 'reagent_1d_smiles' in self.dataset[index].keys():
            data = self.dataset[index]
            
            reactant_list = data['reactant_3d_smiles']
            reagent_list = data['reagent_3d_smiles']
            product_list = [data['product_3d_smiles']]

            for i in range(len(reactant_list)):
                if random.random() < self.mask_mol_ratio:
                    reactant_list[i] = self.tokenizer.mask_token
            for i in range(len(reagent_list)):
                if random.random() < self.mask_mol_ratio:
                    reagent_list[i] = self.tokenizer.mask_token
            for i in range(len(product_list)):
                if random.random() < self.mask_mol_ratio:
                    product_list[i] = self.tokenizer.mask_token

            source_str = ".".join(reactant_list) + '>' + ".".join(reagent_list) + '-->>>>' + ".".join(product_list)    
            target_str = ".".join(data['reactant_1d_smiles']) + '>' + ".".join(data['reagent_1d_smiles']) + '-->>>>' + ".".join([data['product_1d_smiles']])

            source_sample = self.tokenizer(
                            self.prefix+source_str,
                            max_length=self.max_length,
                            padding="do_not_pad",
                            truncation=True,
                            return_tensors='pt',
                        )

            target_sample = self.tokenizer(
                            target_str,
                            max_length=self.max_length,
                            padding="do_not_pad",
                            truncation=True,
                            return_tensors='pt',
                        )

            return [source_sample['input_ids'].squeeze(0), target_sample['input_ids'].squeeze(0)]

        raw_str = self.dataset[index]['smi']
        
        sample = self.tokenizer(
                        self.prefix+raw_str,
                        max_length=self.max_length,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )

        return [sample['input_ids'].squeeze(0)]

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

    def collater(self, items):
        batched_data = self.data_collator([item[0] for item in items])
        samples = [item[-1] for item in items]
        batched_data['labels'] = pad_sequence(samples, batch_first=True, padding_value=self.tokenizer._convert_token_to_id(self.tokenizer.pad_token))
        batched_data['attention_mask'] = batched_data["input_ids"].ne(self.tokenizer._convert_token_to_id(self.tokenizer.pad_token))
        return batched_data


class TaskPrefixDataset(BaseWrapperDataset):
    def __init__(self, dataset, vocab_path, prefix, task_type, max_source_length, max_target_length):
        self.dataset = dataset
        self.prefix = prefix
        self.task_type = task_type
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = SimpleTokenizer(vocab_file=vocab_path)
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        if isinstance(self.dataset[index], dict):
            data = self.dataset[index]
            if 'source' in data.keys():
                raw_str = data['source'] + "-->>>>" + data['target']
                prefix = self.prefix
            else:
                if self.prefix != 'Mixed':
                    prefix = self.prefix
                else:
                    prefix_list = ['Product', 'Reactants', 'Reagents']
                    if data['reagent_1d_smiles']:
                        prefix_idx = random.randint(1, 3)
                    else:
                        prefix_idx = random.randint(1, 2)
                    prefix = prefix_list[prefix_idx-1]
                if prefix == 'Product':
                    raw_str = ".".join(data['reagent_3d_smiles']) + '>' + ".".join(data['reactant_3d_smiles']) + '-->>>>' + ".".join([data['product_1d_smiles']])
                elif prefix == 'Reactants':
                    raw_str = ".".join([data['product_3d_smiles']]) + '-->>>>' + ".".join(data['reactant_1d_smiles'])
                elif prefix == 'Reagents':
                    raw_str = ".".join(data['reactant_3d_smiles']) + '>' + ".".join([data['product_3d_smiles']]) + '-->>>>' + ".".join(data['reagent_1d_smiles'])
        else:
            raw_str = self.dataset[index]
            prefix = self.prefix
        source_str, target_str = raw_str.split("-->>>>")

        if self.task_type == 'forward-mixed':
            reagents, reactants = source_str.split(">")
            if reagents != '':
                source_str = (reactants + '.' + reagents).split('.')
            else:
                source_str = reactants.split('.')
            random.shuffle(source_str)
            source_str = '.'.join(source_str)

        source_sample = self.tokenizer(
                prefix+source_str,
                max_length=self.max_source_length,
                padding="do_not_pad",
                truncation=True,
                return_tensors='pt',
            )
        if self.prefix == 'Yield:':
            target_value = float(target_str)
            target_ids = torch.Tensor([target_value])
        else:
            target_sample = self.tokenizer(
                    target_str,
                    max_length=self.max_target_length,
                    padding="do_not_pad",
                    truncation=True,
                    return_tensors='pt',
                )
            target_ids = target_sample["input_ids"].squeeze(0)

        return {"input_ids": source_sample["input_ids"].squeeze(0), 
                "attention_mask": source_sample["attention_mask"].squeeze(0),
                "decoder_input_ids": target_ids}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
 
    def collater(self, items):
        batched_data = {}
        for key in items[0].keys():
            if 'smiles' in key:
                batched_data[key] = [item[key] for item in items]
            else:
                samples = [item[key] for item in items]
                if 'mask' in key:
                    padding_value = 0
                else:
                    padding_value = self.tokenizer._convert_token_to_id(self.tokenizer.pad_token)
                batched_data[key] = pad_sequence(samples, batch_first=True, padding_value=padding_value)
        source_ids, source_mask, y = batched_data["input_ids"].long(), batched_data["attention_mask"], batched_data["decoder_input_ids"].long()
        if 'smiles' in items[0].keys():
            return {'input_ids': source_ids, 'attention_mask': source_mask, 'labels': y, 'smiles': batched_data["smiles"]}
        else:
            return {'input_ids': source_ids, 'attention_mask': source_mask, 'labels': y}