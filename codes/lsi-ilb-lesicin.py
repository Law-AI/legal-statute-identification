
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import numpy as np
import os
import pickle as pkl
import sys
import json
import transformers

from collections import defaultdict
from dataclasses import dataclass
from datasets import Features, Sequence, load_dataset
from datasets.features import ClassLabel, Value
from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding, TrainingArguments, Trainer, AdamW
from transformers.data.data_collator import DataCollatorMixin
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List
from safetensors.torch import load_model
from safetensors import safe_open


def generate_graph(label_vocab, type_map, label_tree_edges, cit_net_edges, label_name='section'):
    node_vocab = defaultdict(dict) # each key is a node_type, and each value is a dict storing the vocab of all nodes under given node type
    node_vocab[label_name] = label_vocab # manually set this since we want the label vocab to be consistent with node vocab for labels

    edge_vocab = {}
    edge_indices = defaultdict(list) # each key is a tuple (src node type, relationship name, trg node type), and each value is a list storing the edges from src node type to trg node type

    for (node_a, edge_type, node_b) in label_tree_edges + cit_net_edges:
        # first get the node type
        node_a_type, node_b_type = type_map[node_a], type_map[node_b]

        # create new vocab entries for edges and nodes
        if edge_type not in edge_vocab:
            edge_vocab[edge_type] = len(edge_vocab)

        if node_a not in node_vocab[node_a_type]:
            node_vocab[node_a_type][node_a] = len(node_vocab[node_a_type])
        if node_b not in node_vocab[node_b_type]:
            node_vocab[node_b_type][node_b] = len(node_vocab[node_b_type])
            
        # get node indices    
        node_a_token = node_vocab[node_a_type][node_a]
        node_b_token = node_vocab[node_b_type][node_b]

        edge_indices[(node_a_type, edge_type, node_b_type)].append([node_a_token, node_b_token])

    num_nodes = {ntype: len(nodes) for ntype, nodes in node_vocab.items()}

    # same as edge_indices except that the edges under each key are now stored as sparse matrices
    adjacency = {}
    for keys, edges in edge_indices.items():
        row, col = torch.tensor(edges).t()
        sizes = (num_nodes[keys[0]], num_nodes[keys[-1]])
        adj = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=sizes)
        adjacency[tuple(keys)] = adj

    return node_vocab, edge_vocab, edge_indices, adjacency

class LstmAttn(nn.Module):
    def __init__(self, hidden_size, drop=0.5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Parameter(torch.rand(hidden_size))
        self.dropout = nn.Dropout(drop)
        
    def forward(self, inputs=None, attention_mask=None, dynamic_context=None, use_rnn=True):
        if attention_mask is None:
            attention_mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        
        if use_rnn:
            lengths = attention_mask.float().sum(dim=1)
            inputs_packed = pack_padded_sequence(inputs, torch.clamp(lengths, min=1).cpu(), enforce_sorted=False, batch_first=True)  
            outputs_packed = self.lstm(inputs_packed)[0]
            outputs = pad_packed_sequence(outputs_packed, batch_first=True)[0]
        else:
            outputs = inputs

        activated_outputs = torch.tanh(self.dropout(self.attn_fc(outputs)))
        context = dynamic_context if dynamic_context is not None else self.context.expand(inputs.size(0), self.hidden_size)
        scores = torch.bmm(activated_outputs, context.unsqueeze(2)).squeeze(2)
        masked_scores = scores.masked_fill(~attention_mask, -1e-32)
        masked_scores = F.softmax(masked_scores, dim=1)
        
        hidden = torch.sum(outputs * masked_scores.unsqueeze(2), dim=1)
        return outputs, hidden


class HierBert(nn.Module):
    def __init__(self, encoder, fragment_size=None, drop=0.5):
        super().__init__()
        
        self.bert_encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.fragment_size = fragment_size
        self.segment_encoder = LstmAttn(self.hidden_size, drop=drop)
        self.dropout = nn.Dropout(drop)
   
    def gradient_checkpointing_enable(self):
        self.bert_encoder.gradient_checkpointing_enable()

    def _encoder_forward(self, input_ids, attention_mask, dummy):
        return self.dropout(self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)[:, 0, :]
    
    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None, dynamic_context=None):
        if input_ids is not None:
            batch_size, max_num_segments, max_segment_size = input_ids.shape
        else:
            batch_size, max_num_segments = encoder_outputs.shape[:2]
        
        ## encode individual segments using Bert
        if input_ids is not None:
            input_ids_flat = input_ids.view(-1, max_segment_size)
            attention_mask_flat = attention_mask.view(-1, max_segment_size)
            dummy = torch.ones(1, dtype=torch.float, requires_grad=True)
            
            if self.fragment_size is not None:
                encoder_outputs = []
                
                for fragment_idx, sent_idx in enumerate(range(0, batch_size * max_num_segments, self.fragment_size)):
                    input_ids_fragment = input_ids_flat[sent_idx : sent_idx + self.fragment_size]
                    attention_mask_fragment = attention_mask_flat[sent_idx : sent_idx + self.fragment_size]   
                    encoder_outputs_fragment = checkpoint(self._encoder_forward, input_ids_fragment, attention_mask_fragment, dummy, use_reentrant=False)
                    encoder_outputs.append(encoder_outputs_fragment)
                
                encoder_outputs = torch.cat(encoder_outputs, dim=0)
                
            else:
                encoder_outputs = self._encoder_forward(input_ids_flat, attention_mask_flat, dummy)
                
            encoder_outputs = encoder_outputs.view(batch_size, max_num_segments, self.hidden_size)
            attention_mask = attention_mask.any(dim=2)
            
        ## encode each example by aggregating Bert segment outputs
        outputs, hidden = self.segment_encoder(inputs=encoder_outputs, attention_mask=attention_mask, dynamic_context=dynamic_context)
        return outputs, hidden

class MetapathAggrNet(torch.nn.Module):
    def __init__(self, node_vocab_size, edge_vocab_size, hidden_size, drop=0.1, gdel=14.):
        super().__init__()
        self.emb_range = gdel / hidden_size
        
        self.node_embedding = torch.nn.ModuleDict({ntype: torch.nn.Embedding(num_nodes, hidden_size) for ntype, num_nodes in node_vocab_size.items()})
        for ntype, ntype_weights in self.node_embedding.items():
            ntype_weights.weight.data.uniform_(- self.emb_range, self.emb_range)
        
        self.scale_fc = torch.nn.ModuleDict({ntype: torch.nn.Linear(hidden_size, hidden_size) for ntype in node_vocab_size})
        
        self.edge_embedding = torch.nn.Embedding(edge_vocab_size, hidden_size // 2)
        self.edge_embedding.weight.data.uniform_(- self.emb_range, self.emb_range)
        
        self.intra_attention = LstmAttn(2 * hidden_size, drop=drop)
        
        self.inter_fc = torch.nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.inter_context = torch.nn.Parameter(torch.rand(2 * hidden_size))
        
        self.output_fc = torch.nn.Linear(2 * hidden_size, hidden_size)
        
        self.dropout = torch.nn.Dropout(drop)
    
    # Embed each node index using the node embedding matrix and then scale to generate same sized embeddings for each node type     
    def embed_and_scale(self, tokens, edge_tokens, schema): # [B, L+1], [B, L]
        inputs, edge_inputs = [], []
        
        node_type = schema[0][0]  
        node_input = self.dropout(self.node_embedding[node_type](tokens[:, 0])) # [B, H]
        inputs.append(self.dropout(self.scale_fc[node_type](node_input)))
        
        for i in range(edge_tokens.size(1)):
            node_type = schema[i][2]
            node_input = self.dropout(self.node_embedding[node_type](tokens[:, i+1])) # [B, H]
            inputs.append(self.dropout(self.scale_fc[node_type](node_input)))
                          
            edge_inputs.append(self.dropout(self.edge_embedding(edge_tokens[:, i])))
        inputs = torch.stack(inputs, dim=1) # [B, L+1, H]
        edge_inputs = torch.stack(edge_inputs, dim=1) # [B, L, H]
        return inputs, edge_inputs
    
    # We are following the official implementation of the RotatE algorithm --- https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
    def rotational_encoding(self, inputs, edge_inputs): # [B, L+1, H], [B, L, H/2]
        PI = 3.14159265358979323846
        hidden = inputs.clone()
        for i in reversed(range(edge_inputs.size(1))):
            hid_real, hid_imag = torch.chunk(hidden.clone()[:, i+1:, :], 2, dim=2) # [B, L-i, H/2], [B, L-i, H/2]
            inp_real, inp_imag = torch.chunk(inputs[:, i, :], 2, dim=1) # [B, H/2], [B, H/2]
            
            edge_complex = edge_inputs[:, i, :] / (self.emb_range / PI)
            edge_real, edge_imag = torch.cos(edge_inputs[:, i, :]), torch.sin(edge_inputs[:, i, :]) # [B, H/2], [B, H/2]
           
            out_real = inp_real.unsqueeze(1) + edge_real.unsqueeze(1) * hid_real - edge_imag.unsqueeze(1) * hid_imag # [B, L-i, H/2]
            out_imag = inp_imag.unsqueeze(1) + edge_imag.unsqueeze(1) * hid_real + edge_real.unsqueeze(1) * hid_imag # [B, L-i, H/2]
            
            hidden[:, i+1:, :] = torch.cat([out_real, out_imag], dim=2)
        path_lens = 1 + torch.arange(hidden.size(1), device=hidden.device) # [L+1]
        return hidden / path_lens.unsqueeze(0).unsqueeze(2)
                               
    def forward(self, tokens, edge_tokens, schemas, intra_context=None, inter_context=None): 
        hidden = []

        # serially perform intra-metapath aggregation across the different schemas
        for i in range(len(tokens)):
            # flatten out the multiple samples of the same schema                       
            mpath_tokens = tokens[i].view(-1, tokens[i].size(2)) # [M*D, L+1]
            mpath_edge_tokens = edge_tokens[i].view(-1, edge_tokens[i].size(2)) # [M*D, L]
                               
            mpath_inputs, mpath_edge_inputs = self.embed_and_scale(mpath_tokens, mpath_edge_tokens, schemas[i])
                               
            mpath_hidden_all = self.rotational_encoding(mpath_inputs, mpath_edge_inputs) # [M*D, L+1, H]

            # the first element in the sequence is the target node, the rest are transformed embeddings for other nodes in the metapath
            mpath_hidden_all = torch.cat([mpath_hidden_all[:, 0, :].unsqueeze(1).repeat(1, mpath_hidden_all.size(1) - 1, 1), mpath_hidden_all[:, 1:, :]], dim=2) # [M*D, L, 2H]                   
            mpath_hidden = torch.relu(self.intra_attention(mpath_hidden_all, dynamic_context=intra_context, use_rnn=False)[1]) # [M*D, 2H]

            # aggregate transformed embeddings from multiple samples of the same schema 
            mpath_hidden = torch.sum(mpath_hidden.view(tokens[i].size(0), tokens[i].size(1), -1), dim=0) # [D, 2H]
            hidden.append(mpath_hidden)
        hidden = torch.stack(hidden, dim=1) # [D, N, 2H]
        
        # perform inter-metapath aggregation across transformed embeddings for each schema
        hidden_act = torch.mean(torch.tanh(self.dropout(self.inter_fc(hidden))), dim=0).expand_as(hidden) # [D, N, 2H]
        context = self.inter_context.unsqueeze(0).repeat(hidden_act.size(0), 1).unsqueeze(2) if inter_context is None else inter_context.unsqueeze(2)
        scores = torch.bmm(hidden_act, context) # [D, N, 1]
                               
        outputs = torch.sum(hidden * scores, dim=1) # [D, 2H]
        outputs = self.dropout(self.output_fc(outputs)) # [D, H]
        
        return outputs

class MatchNet(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, drop=0.1):
        super().__init__()
        
        self.matcher = LstmAttn(hidden_size, drop=drop)
        self.match_fc = torch.nn.Linear(2 * hidden_size, num_labels)
        
        self.dropout = torch.nn.Dropout(drop)
        
    def forward(self, fact_inputs, sec_inputs, context=None): # [D, H], [C, H]
        sec_inputs = sec_inputs.expand(fact_inputs.size(0), sec_inputs.size(0), sec_inputs.size(1)) # [D, C, H]
        
        sec_hidden = self.matcher(sec_inputs, dynamic_context=context)[1]
        
        logits = self.dropout(self.match_fc(torch.cat([fact_inputs, sec_hidden], dim=1))) # [D, C]
        scores = torch.sigmoid(logits).detach() # [D, C]
        return logits, scores

@dataclass
class TextClassifierOutput(ModelOutput):
    loss:torch.Tensor = None
    logits:torch.Tensor = None
    hidden_states:torch.Tensor = None

# Complete LeSICiN model
class LeSICiNBertForTextClassification(torch.nn.Module):
    def __init__(
        self, 
        text_encoder, 
        graph_encoder, 
        match_network, 
        hidden_size, 
        label_weights=None, 
        schemas=None, 
        sec_schemas=None, 
        lambdas=(0.5, 0.5), 
        thetas=(3, 2, 3), 
        num_mpaths=8, 
        drop=0.1):
        
        super().__init__()
        
        self.text_encoder = text_encoder
        self.graph_encoder = graph_encoder
        self.match_network = match_network
        
        self.match_context_transform = torch.nn.Linear(hidden_size, hidden_size)
        self.intra_context_transform = torch.nn.Linear(hidden_size, 2 * hidden_size) # We need double the hidden size for Struct Encoder dynamic context
        self.inter_context_transform = torch.nn.Linear(hidden_size, 2 * hidden_size)
        
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=label_weights)
        
        # self.pred_threshold = pthresh
        self.lambdas = lambdas # weights for scores
        self.thetas = thetas # weights for losses
        self.num_mpaths = num_mpaths
        self.schemas = schemas
        self.sec_schemas = sec_schemas
        self.dropout = torch.nn.Dropout(drop)
        
    def calculate_losses(self, logits_list, labels):
        loss = 0
        for i, logits in enumerate(logits_list):
            if logits is not None:
                loss += self.thetas[i] * self.criterion(logits, labels)
        return loss
        
    def forward(
        self,
        ids=None,
        input_ids = None, 
        attention_mask = None, 
        labels = None,
        node_input_ids = None,
        edge_input_ids = None,
        sec_ids = None,
        sec_input_ids = None,
        sec_attention_mask = None,
        sec_node_input_ids = None,
        sec_edge_input_ids = None
    ):

        fact_attr_hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[1]
        
        sec_attr_hidden = self.text_encoder(input_ids=sec_input_ids, attention_mask=sec_attention_mask)[1]
        
        # context vector for matching with fact attributes
        attr_match_context = self.dropout(self.match_context_transform(fact_attr_hidden)) # [D, H]
        
        # Attribute scores
        attr_logits, attr_scores = self.match_network(fact_attr_hidden, sec_attr_hidden, context=attr_match_context) 
        
        # sec-side context vectors for Struct Encoder
        sec_intra_context = self.dropout(self.intra_context_transform(sec_attr_hidden)).repeat(self.num_mpaths, 1) # [M*C, H]
        sec_inter_context = self.dropout(self.inter_context_transform(sec_attr_hidden)) # [C, H]
        
        # Encode sec graph using MAGNN
        sec_struct_hidden = self.graph_encoder(sec_node_input_ids, sec_edge_input_ids, self.sec_schemas, intra_context=sec_intra_context, inter_context=sec_inter_context) # [C, H] 
        
        # Alignment scores
        align_logits, align_scores = self.match_network(fact_attr_hidden, sec_struct_hidden, context=attr_match_context)
        
        if node_input_ids is not None:
            # fact-side context vectors for Struct Encoder
            fact_intra_context = self.dropout(self.intra_context_transform(fact_attr_hidden)).repeat(self.num_mpaths, 1) # [M*D, H]
            fact_inter_context = self.dropout(self.inter_context_transform(fact_attr_hidden)) # [D, H]
            
            # Encode sec graph using MAGNN
            fact_struct_hidden = self.graph_encoder(node_input_ids, edge_input_ids, self.schemas, intra_context=fact_intra_context, inter_context=fact_inter_context) # [D, H]
            
            # context vector for matching with fact structure
            struct_match_context = self.dropout(self.match_context_transform(fact_struct_hidden)) # [D, H]

            # Structural scores
            struct_logits, struct_scores = self.match_network(fact_struct_hidden, sec_struct_hidden, context=struct_match_context)
            
        else:
            struct_logits = None
        
        # Combine scores and losses    
        scores = (self.lambdas[0] * attr_scores + self.lambdas[-1] * align_scores)
        
        loss = self.calculate_losses([attr_logits, struct_logits, align_logits], labels)
        
        return TextClassifierOutput(loss=loss, logits=scores)


def generate_text_tensors(dataset, tokenizer):
    example_ids = []
    
    max_segments = min(max(len(exp['input_ids']) for exp in dataset), 96)
    max_segment_size = min(max(max(len(sent) for sent in exp['input_ids']) for exp in dataset), 128)
    # max_segments, max_segment_size = 128, 128

    input_ids = torch.zeros(len(dataset), max_segments, max_segment_size, dtype=torch.long).fill_(tokenizer.pad_token_id)
    
    for exp_idx, exp in enumerate(dataset):
        example_ids.append(exp['id'])
        for sent_idx, sent in enumerate(exp['input_ids'][:96]):
            sent = sent[:128]
            input_ids[exp_idx, sent_idx, :len(sent)] = sent
    
    attention_mask = input_ids != tokenizer.pad_token_id
    return example_ids, input_ids, attention_mask
   

def generate_metapaths(indices, schemas, adjacency, edge_vocab, num_samples=8): # [D,]
    indices = indices.repeat(num_samples) # [M*D,]
    
    tokens, edge_tokens = [], []
    
    # repeat over all schemas
    for i in range(len(schemas)):
        ins_tokens, ins_edge_tokens = [indices], []
        for keys in schemas[i]:
            neighbours = adjacency[keys].sample(num_neighbors=1, subset=ins_tokens[-1]).squeeze(1) # [M*D,]
            relations = torch.full(neighbours.shape, edge_vocab[keys[1]], dtype=torch.long) # [M*D,]
            
            ins_tokens.append(neighbours)
            ins_edge_tokens.append(relations)
        
        ins_tokens = torch.stack(ins_tokens, dim=1)
        ins_tokens = ins_tokens.view(num_samples, -1, ins_tokens.size(1)) # [M, D, L+1]
        
        ins_edge_tokens = torch.stack(ins_edge_tokens, dim=1)
        ins_edge_tokens = ins_edge_tokens.view(num_samples, -1, ins_edge_tokens.size(1)) # [M, D, L]
        
        tokens.append(ins_tokens)
        edge_tokens.append(ins_edge_tokens)
    
    return tokens, edge_tokens


def collate_sections(sec_dataset, tokenizer, schemas=None, adjacency=None, type_map=None, node_vocab=None, edge_vocab=None):
    
    example_ids = []
    
    max_segments = min(max(len(exp['input_ids']) for exp in sec_dataset), 128)
    max_segment_size = min(max(max(len(sent) for sent in exp['input_ids']) for exp in sec_dataset), 128)
    # max_segments, max_segment_size = 128, 128

    input_ids = torch.zeros(len(sec_dataset), max_segments, max_segment_size, dtype=torch.long).fill_(tokenizer.pad_token_id)
    
    for exp_idx, exp in enumerate(sec_dataset):
        example_ids.append(exp['id'])
        for sent_idx, sent in enumerate(exp['input_ids'][:128]):
            sent = sent[:128]
            input_ids[exp_idx, sent_idx, :len(sent)] = sent
    
    attention_mask = input_ids != tokenizer.pad_token_id

    node_tokens, edge_tokens = None, None
    if schemas is not None:
        trg_node_tokens = torch.tensor([node_vocab[type_map[x]][x] for x in example_ids])
        node_tokens, edge_tokens = generate_metapaths(trg_node_tokens, schemas, adjacency, edge_vocab, num_samples=8)

    
    return BatchEncoding(
        {
            'ids': example_ids,
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'node_input_ids': node_tokens,
            'edge_input_ids': edge_tokens 
        }
    )


@dataclass
class DataCollatorForLSIGraph(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    label_vocab: Dict[str, int]
    # sec_data: datasets.arrow_dataset.Dataset
    sec_ids: List = None
    sec_input_ids: torch.Tensor = None
    sec_attention_mask: torch.Tensor = None 
    return_tensors: str = 'pt'
    schemas: Dict[str, List] = None
    adjacency: torch.Tensor = None
    type_map: Dict[str, str] = None
    node_vocab: Dict[str, int] = None
    edge_vocab: Dict[str, int] = None
    use_graph: bool = True

    def torch_call(self, examples):
        ids, input_ids, attention_mask = generate_text_tensors(examples, self.tokenizer)
        
        labels = torch.zeros(len(examples), len(self.label_vocab))        
        for exp_idx, exp in enumerate(examples):
            labels[exp_idx].scatter_(0, exp['labels'], 1.)

        node_input_ids, edge_input_ids = None, None

        sec_trg_node_tokens = torch.tensor([self.node_vocab[self.type_map[x]][x] for x in self.sec_ids])
        sec_node_input_ids, sec_edge_input_ids = generate_metapaths(sec_trg_node_tokens, self.schemas['section'], self.adjacency, self.edge_vocab, num_samples=8)

        if self.use_graph:
            trg_node_tokens = torch.tensor([self.node_vocab[self.type_map[x]][x] for x in ids])
            node_input_ids, edge_input_ids = generate_metapaths(trg_node_tokens, self.schemas['fact'], self.adjacency, self.edge_vocab, num_samples=8)
        
        return BatchEncoding(
            {
                'ids': ids,
                'input_ids': input_ids, 
                'attention_mask': attention_mask, 
                'labels': labels,
                'node_input_ids': node_input_ids,
                'edge_input_ids': edge_input_ids,
                'sec_ids': self.sec_ids,
                'sec_input_ids': self.sec_input_ids,
                'sec_attention_mask': self.sec_attention_mask,
                'sec_node_input_ids': sec_node_input_ids,
                'sec_edge_input_ids': sec_edge_input_ids
            }
        )

# Compute macro F1 scores
def compute_metrics(p, threshold=0.75):
    metrics = {}
    preds = (p.predictions > threshold).astype(float)
    refs = p.label_ids
    metrics['prec'] = precision_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['rec'] = recall_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['f1'] = f1_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    return metrics

# Different layers have different learning rates, here we set the learning rates for each layer
def AdamWLLRD(model, bert_lr=3e-5, intermediate_lr=1e-3, top_lr=1e-4, wd=1e-2):
    opt_params = []
    named_params = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert = ["text_encoder.bert_encoder.embeddings", "text_encoder.bert_encoder.encoder"]    
    intermediate = ["text_encoder.bert_encoder.pooler", "text_encoder.segment_encoder"]
    top = ["graph_encoder", "match_network", "_context_transform"]

    ### Top layers
    pnd = [p for (n, p) in named_params if any(t in n for t in top) and any(nd in n for nd in no_decay)]
    pd = [p for (n, p) in named_params if any(t in n for t in top) and not any(nd in n for nd in no_decay)]

    top_params = {"params": pnd, "lr": top_lr, "weight_decay": wd}
    opt_params.append(top_params)
    top_params = {"params": pd, "lr": top_lr, "weight_decay": wd}
    opt_params.append(top_params)
    
    ### Intermediate layers
    pnd = [p for (n, p) in named_params if any(i in n for i in intermediate) and any(nd in n for nd in no_decay)]
    pd = [p for (n, p) in named_params if any(i in n for i in intermediate) and not any(nd in n for nd in no_decay)]

    intermediate_params = {"params": pnd, "lr": intermediate_lr, "weight_decay": wd}
    opt_params.append(intermediate_params)
    intermediate_params = {"params": pd, "lr": intermediate_lr, "weight_decay": wd}
    opt_params.append(intermediate_params)
    
    ### Bert layers
    pnd = [p for (n, p) in named_params if any(b in n for b in bert) and any(nd in n for nd in no_decay)]
    pd = [p for (n, p) in named_params if any(b in n for b in bert) and not any(nd in n for nd in no_decay)]

    bert_params = {"params": pnd, "lr": bert_lr, "weight_decay": 0}
    opt_params.append(bert_params)
    bert_params = {"params": pd, "lr": bert_lr, "weight_decay": wd}
    opt_params.append(bert_params)

    return AdamW(opt_params, lr=bert_lr, correct_bias=True)

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


class LSIGraphTrainer(Trainer):
    def __init__(self, label_vocab=None, sec_ids=None, sec_input_ids=None, sec_attention_mask=None, schemas=None, adjacency=None, type_map=None, node_vocab=None, edge_vocab=None, **kwargs):
        super().__init__(**kwargs)

        self.label_vocab = label_vocab
        self.sec_ids = sec_ids
        self.sec_input_ids = sec_input_ids
        self.sec_attention_mask = sec_attention_mask
        self.schemas = schemas
        self.adjacency = adjacency
        self.type_map = type_map
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab

    def get_train_dataloader(self):
        train_collator = DataCollatorForLSIGraph(
            tokenizer=self.tokenizer, 
            label_vocab=self.label_vocab, 
            sec_ids=self.sec_ids,
            sec_input_ids=self.sec_input_ids,
            sec_attention_mask=self.sec_attention_mask,
            schemas=self.schemas,
            adjacency=self.adjacency,
            type_map=self.type_map,
            node_vocab=self.node_vocab,
            edge_vocab=self.edge_vocab,
            use_graph=True
        )
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.train_batch_size, 
            sampler=self._get_train_sampler(), 
            collate_fn=train_collator, 
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
            prefetch_factor=8)

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_collator = DataCollatorForLSIGraph(
            tokenizer=self.tokenizer, 
            label_vocab=self.label_vocab, 
            sec_ids=self.sec_ids,
            sec_input_ids=self.sec_input_ids,
            sec_attention_mask=self.sec_attention_mask,
            schemas=self.schemas,
            adjacency=self.adjacency,
            type_map=self.type_map,
            node_vocab=self.node_vocab,
            edge_vocab=self.edge_vocab,
            use_graph=False
        )
        return DataLoader(
            eval_dataset, 
            batch_size=self.args.eval_batch_size, 
            sampler=self._get_eval_sampler(eval_dataset), 
            collate_fn=eval_collator, 
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
    def get_test_dataloader(self, test_dataset=None):
        test_dataset = test_dataset if test_dataset is not None else self.test_dataset
        test_collator = DataCollatorForLSIGraph(
            tokenizer=self.tokenizer, 
            label_vocab=self.label_vocab, 
            sec_ids=self.sec_ids,
            sec_input_ids=self.sec_input_ids,
            sec_attention_mask=self.sec_attention_mask,
            schemas=self.schemas,
            adjacency=self.adjacency,
            type_map=self.type_map,
            node_vocab=self.node_vocab,
            edge_vocab=self.edge_vocab,
            use_graph=False
        )
        return DataLoader(
            test_dataset, 
            batch_size=self.args.eval_batch_size, 
            sampler=self._get_eval_sampler(test_dataset), 
            collate_fn=test_collator, 
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )



if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = 'true'
    
    root = sys.argv[1]              # Dataset directory
    model_src = sys.argv[2]         # Model name/directory
    output_fol = sys.argv[3]        # Output Folder

    with open(os.path.join(root, "label_vocab.json")) as fr:
        label_vocab = json.load(fr)


    schema = Features(
        {
            "id": Value('string'),
            "text": Sequence(Value('string')),
            "labels": Sequence(
                ClassLabel(num_classes=len(label_vocab), names=list(label_vocab.keys()))
            )
        }
    )

    sec_schema = Features(
        {
            "id": Value('string'),
            "text": Sequence(Value('string'))
        }
    )

    dataset = load_dataset(
        'json', 
        data_files={'train': os.path.join(root, "train2.json"), 
                    'dev': os.path.join(root, "dev2.json"), 
                    'test': os.path.join(root, "test2-expln3-noimp.json")}, 
        field='data', 
        cache_dir='Cache')
    print(dataset)

    sec_dataset = load_dataset(
        'json',
        data_files={'main': os.path.join(root, "secs.json")},
        field='data',
        cache_dir='Cache'
    )
    print(sec_dataset)

    dataset = dataset.map(schema.encode_example, features=schema)
    sec_dataset = sec_dataset.map(sec_schema.encode_example, features=sec_schema)
    print(type(dataset['train']))

    tokenizer = AutoTokenizer.from_pretrained(model_src, cache_dir='Cache')

    dataset = dataset.map(lambda example: tokenizer(list(example['text']), return_token_type_ids=False), batched=False)
    sec_dataset = sec_dataset.map(lambda example: tokenizer(list(example['text']), return_token_type_ids=False), batched=False)

    if not os.path.exists(os.path.join(root, "label_weights_custom.pkl")):
        label_weights = torch.zeros(len(label_vocab))
        for exp in tqdm(dataset['train']):
            for l in exp['labels']:
                label_weights[l] += 1
        label_weights = torch.clamp(label_weights.max() / label_weights, max=10.0)
        with open(os.path.join(root, "label_weights_custom.pkl"), 'wb') as fw:
            pkl.dump(label_weights, fw)
    else:
        with open(os.path.join(root, "label_weights_custom.pkl"), 'rb') as fr:
            label_weights = pkl.load(fr)

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
    sec_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    

    with open(os.path.join(root, "type_map.json")) as fr:
        type_map = json.load(fr)
    with open(os.path.join(root, "label_tree.json")) as fr:
        label_tree = json.load(fr)
    with open(os.path.join(root, "citation_network.json")) as fr:
        citation_net = json.load(fr)
    with open(os.path.join(root, "schemas.json")) as fr:
        schemas = json.load(fr)
    for sch in schemas.values():
        for path in sch:
            for i, edge in enumerate(path):
                path[i] = tuple(path[i])

    node_vocab, edge_vocab, edge_indices, adjacency = generate_graph(label_vocab, type_map, label_tree, citation_net) 

    L = len(label_vocab)
    N = {k: len(v) for k,v in node_vocab.items()}
    E = len(edge_vocab)

    bert = AutoModel.from_pretrained(model_src, cache_dir='Cache')
    text_encoder = HierBert(bert)
    text_encoder.gradient_checkpointing_enable()
    graph_encoder = MetapathAggrNet(N, E, 768)
    match_network = MatchNet(768, L)

    model = LeSICiNBertForTextClassification(
        text_encoder,
        graph_encoder,
        match_network,
        768,
        label_weights=label_weights,
        schemas=schemas['fact'],
        sec_schemas=schemas['section'],
    )
    
    tensors = dict()
    with safe_open(os.path.join(output_fol, "model.safetensors"), framework='pt', device='cuda') as fr:
        for k in fr.keys():
            tensors[k] = fr.get_tensor(k)
    model.load_state_dict(tensors)

    opt = AdamWLLRD(model)
    sch = transformers.get_constant_schedule_with_warmup(opt, num_warmup_steps=1000)

    training_args = TrainingArguments(
        output_dir=output_fol,
        overwrite_output_dir=False,
        do_train=False,
        do_eval=True,
        do_predict=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=None,
        learning_rate=None,
        weight_decay=None,
        num_train_epochs=15,
        logging_strategy='epoch',
        logging_first_step=True,
        save_strategy='epoch',
        save_total_limit=3,
        seed=42,
        fp16=True,
        dataloader_num_workers=16,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        group_by_length=False,
        dataloader_pin_memory=True,
        resume_from_checkpoint=True,
        gradient_checkpointing=False,
    )

    sec_ids, sec_input_ids, sec_attention_mask = generate_text_tensors(sec_dataset['main'], tokenizer)
    
    trainer = LSIGraphTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        compute_metrics=compute_metrics,
        optimizers=(opt, sch),
        label_vocab=label_vocab,
        sec_ids=sec_ids,
        sec_input_ids=sec_input_ids,
        sec_attention_mask=sec_attention_mask,
        schemas=schemas,
        adjacency=adjacency,
        type_map=type_map,
        node_vocab=node_vocab,
        edge_vocab=edge_vocab
    )

    if training_args.do_train:
        _, _, metrics = trainer.train(ignore_keys_for_eval=['hidden_states'], resume_from_checkpoint=False)
        trainer.save_model()
        trainer.save_metrics('train', metrics)
    if training_args.do_eval:
        test_results = trainer.evaluate(eval_dataset=dataset['test'], ignore_keys=['hidden_states'])
        print(test_results)
        trainer.save_metrics('test', test_results) #['eval_metrics'])
    if training_args.do_predict:
        predictions, label_ids, results = trainer.predict(test_dataset=dataset['test'], ignore_keys=['hidden_states'])
        np.save(os.path.join(output_fol, "predictions.npy"), predictions)
        np.save(os.path.join(output_fol, "label_ids.npy"), label_ids)
        trainer.save_metrics('test', results)
        print(results)
