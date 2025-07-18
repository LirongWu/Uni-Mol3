# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicore import utils
from unimol3.data import SimpleTokenizer
from transformers import T5ForConditionalGeneration, T5Config
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture

logger = logging.getLogger(__name__)


@register_model("T5")
class T5Model(BaseUnicoreModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--load-pretrain", type=bool, help="whether to load a pre-trained model"
        )


    def __init__(self, args, dictionary=None):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.regression_heads = nn.ModuleDict()
        self.tokenizer = SimpleTokenizer(vocab_file=args.data + "vocab.pt")

        if not args.load_pretrain:
            config = T5Config(
                vocab_size=len(self.tokenizer),
                num_layers=args.encoder_layers,
                num_heads=args.encoder_attention_heads, 
                d_model=args.encoder_embed_dim,
                decoder_start_token_id=self.tokenizer._convert_token_to_id(self.tokenizer.bos_token),
                eos_token_id=self.tokenizer._convert_token_to_id(self.tokenizer.eos_token),
                pad_token_id=self.tokenizer._convert_token_to_id(self.tokenizer.pad_token),
                output_past=True,
            )
            self.T5Model = T5ForConditionalGeneration(config)
            print("Training From Scratch ...")
        else:
            self.T5Model = T5ForConditionalGeneration.from_pretrained("./model_weights/simple/")
            print("Loading Pre-trained Model ...")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        regression=False,
        **kwargs
    ):
        if regression is True:
            encoder_outputs = self.T5Model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            outputs = encoder_outputs.last_hidden_state
            logits = self.regression_heads['prediction_head'](outputs)
            
            loss_fct = nn.KLDivLoss(reduction='batchmean')
            smoothed_label = torch.stack([(100-labels), labels], dim=1)/100
            logits = nn.functional.log_softmax(logits, dim=-1)
            loss = loss_fct(logits, smoothed_label.view(-1,2))
            logits = torch.exp(logits[:,-1])*100
            return {"loss": loss, "logits": logits}
        else:
            outputs = self.T5Model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs

    def register_regression_head(self, inner_dim=None, **kwargs):
        """Register a regression head."""
        self.regression_heads['prediction_head'] = RegressionHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.encoder_embed_dim,
            num_classes=2,
            activation_fn='relu',
            pooler_dropout=0.1,
        )

class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("T5", "T5")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.load_pretrain = getattr(args, "load_pretrain", False)


@register_model_architecture("T5", "T5_base")
def bert_base_architecture(args):
    base_architecture(args)


@register_model_architecture("T5", "T5_large")
def unimol_base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)


@register_model_architecture("T5", "T5_pretrain")
def unimol_base_architecture(args):
    args.load_pretrain = getattr(args, "load_pretrain", True)
    base_architecture(args)