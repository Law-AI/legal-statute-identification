
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle as pkl
import sys
import json
import transformers

from datasets import Features, Sequence, load_dataset
from datasets.features import ClassLabel, Value
from stas.model_fast import *

from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import BatchEncoding, TrainingArguments, Trainer, AdamW
from transformers.file_utils import ModelOutput
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

DEVICE = 'cuda'

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

root = sys.argv[1]              # Dataset directory
model_src = sys.argv[2]         # Model name/directory
output_fol = sys.argv[3]        # Output Folder

with open(os.path.join(root, "label_vocab.json")) as fr:
    label_vocab = json.load(fr)

with open(os.path.join(root, "label2community.json")) as fr:
    label2community = json.load(fr)
    if type(list(label2community.keys())[0]) == str:
        label2community = {int(k): v for k,v in label2community.items()}
    
num_communities = len(set(label2community.values()))
print(num_communities)

schema = Features(
    {
        "id": Value('string'),
        "text": Sequence(Value('string')),
        "labels": Sequence(
            ClassLabel(num_classes=len(label_vocab), names=list(label_vocab.keys()))
        )
    }
)

label_schema = Features(
    {
        "id": Value('string'),
        "title": Value('string'),
        "text": Sequence(Value('string'))
    }
)

dataset = load_dataset('json', data_files={'train': os.path.join(root, "train2.json"), 
                                           'dev': os.path.join(root, "dev2.json"), 
                                           'test': os.path.join(root, "test2.json")}, field='data', cache_dir='~/HDD/LSI-Cache')
label_dataset = load_dataset('json', data_files={'label': os.path.join(root, "label_descriptions.json")}, field='data', cache_dir='~/HDD/LSI-Cache')


dataset = dataset.map(schema.encode_example, features=schema)
dataset = dataset.filter(lambda example: len(example['text']) != 0)
label_dataset = label_dataset.map(label_schema.encode_example, features=label_schema)

config = AutoConfig.from_pretrained(model_src)
tokenizer = AutoTokenizer.from_pretrained(model_src) #, cache_dir='~/HDD/LSI-Cache')
dataset = dataset.map(lambda example: tokenizer(list(example['text']), return_token_type_ids=False), batched=False)
label_dataset = label_dataset.map(lambda example: tokenizer([example['title'] + ": "] + list(example['text']), return_token_type_ids=False), batched=False)

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
label_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Generate community weights in the reverse order of their frequency
def generate_community_weights():
    weights = torch.zeros(num_communities)
    for exp in tqdm(dataset['train'], desc="Label Weights"):
        _weights = torch.zeros(num_communities)
        for l in exp['labels']:
            _weights[label2community[l.item()]] = 1
        weights += _weights
    return len(dataset['train']) / weights

if os.path.exists(os.path.join(root, 'community_weights.pkl')):
    with open(os.path.join(root, 'community_weights.pkl'), 'rb') as fr:
        community_weights = pkl.load(fr)
else:
    community_weights = generate_community_weights() #.to(DEVICE)
    with open(os.path.join(root, 'community_weights.pkl'), 'wb') as fw:
        pkl.dump(community_weights, fw)

# Generate label weights in the reverse order of their frequency
def generate_label_weights():
    weights = torch.zeros(len(label_vocab))
    for exp in tqdm(dataset['train'], desc="Label Weights"):
        for l in exp['labels']:
            weights[l] += 1
    return len(dataset['train']) / weights

if os.path.exists(os.path.join(root, 'label_weights.pkl')):
    with open(os.path.join(root, 'label_weights.pkl'), 'rb') as fr:
        label_weights = pkl.load(fr)
else:
    label_weights = generate_label_weights() #.to(DEVICE)
    with open(os.path.join(root, 'label_weights.pkl'), 'wb') as fw:
        pkl.dump(label_weights, fw)
        
community2label = torch.zeros(num_communities, len(label_vocab)) #.to(DEVICE)
for l,c in label2community.items():
    community2label[c,l] = 1
    
with open(os.path.join(root, 'ladan_edges.json')) as fr:
    edges = json.load(fr)
edges = torch.tensor(edges).permute(1, 0) #.to(DEVICE) # [2, num edges]

# Data collator
def collate_fn(examples):
    haslabels = True if 'labels' in examples[0] else False
    
    max_segments = min(max(len(exp['input_ids']) for exp in examples), 128)
    max_segment_size = min(max(max(len(sent) for sent in exp['input_ids']) for exp in examples), 128)
    
    input_ids = torch.zeros(len(examples), max_segments, max_segment_size, dtype=torch.long).fill_(tokenizer.pad_token_id)
    if haslabels:
        labels = torch.zeros(len(examples), len(label_vocab))
        community_labels = torch.zeros(len(examples), num_communities)
    
    for exp_idx, exp in enumerate(examples):
        for sent_idx, sent in enumerate(exp['input_ids'][:128]):
            sent = sent[:128]
            input_ids[exp_idx, sent_idx, :len(sent)] = sent
        if haslabels:
            labels[exp_idx].scatter_(0, exp['labels'], 1.)
            comms = torch.tensor(list(set(label2community[l.item()] for l in exp['labels'])), dtype=torch.long)
            community_labels[exp_idx].scatter_(0, comms, 1.)
        
    attention_mask = input_ids != tokenizer.pad_token_id
    
    
    if haslabels:
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask, 'label_input_ids': label_batch.input_ids, 'label_attention_mask': label_batch.attention_mask, 'community2label': community2label, 'edges': edges, 'labels': labels, 'community_labels': community_labels})
    else:
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})


print("Encoding labels ...", end= ' ')
label_loader = torch.utils.data.DataLoader(label_dataset['label'], batch_size=len(label_vocab), collate_fn=collate_fn)
for label_batch in label_loader:
    pass
print(label_batch.input_ids.shape)
print("Done!")

class LstmAttn(nn.Module):
    def __init__(self, hidden_size, drop=0.5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Parameter(torch.rand(hidden_size))
        self.dropout = nn.Dropout(drop)
        
    def forward(self, inputs=None, attention_mask=None, dynamic_context=None):
        if attention_mask is None:
            attention_mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        
        
        lengths = attention_mask.float().sum(dim=1)
        
        outputs = self.lstm(inputs)[0]  
        
        activated_outputs = torch.tanh(self.dropout(self.attn_fc(outputs)))
        context = dynamic_context if dynamic_context is not None else self.context
        context = context.expand(inputs.size(0), self.hidden_size)
        scores = torch.bmm(activated_outputs, context.unsqueeze(2)).squeeze(2)
        
        masked_scores = scores.masked_fill(~attention_mask, -1e-32)
        masked_scores = F.softmax(masked_scores, dim=1)
        
        hidden = torch.sum(outputs * masked_scores.unsqueeze(2), dim=1)
        return outputs, hidden
    
class HierBert(nn.Module):
    def __init__(self, encoder, drop=0.5):
        super().__init__()
        
        self.bert_encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.segment_encoder = LstmAttn(self.hidden_size, drop=drop)
        self.dropout = nn.Dropout(drop)
   
    def gradient_checkpointing_enable(self):
        self.bert_encoder.gradient_checkpointing_enable()

    
    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None, segment_size=None):
        if input_ids is not None:
            batch_size, max_num_segments, max_segment_size = input_ids.shape
        else:
            batch_size, max_num_segments = encoder_outputs.shape[:2]
        
        ## encode individual segments using Bert
        if input_ids is not None:
            input_ids_flat = input_ids.view(-1, max_segment_size)
            attention_mask_flat = attention_mask.view(-1, max_segment_size)            
            if segment_size is not None:
                encoder_outputs = []
                
                for fragment_idx, sent_idx in enumerate(range(0, batch_size * max_num_segments, segment_size)):
                    input_ids_fragment = input_ids_flat[sent_idx : sent_idx + segment_size]
                    attention_mask_fragment = attention_mask_flat[sent_idx : sent_idx + segment_size]   
                    encoder_outputs_fragment = self.bert_encoder(input_ids=input_ids_fragment, attention_mask=attention_mask_fragment).last_hidden_state[:, 0, :]
                    encoder_outputs.append(encoder_outputs_fragment)
                
                encoder_outputs = torch.cat(encoder_outputs, dim=0)
                
            else:
                encoder_outputs = self.bert_encoder(input_ids_flat, attention_mask_flat).last_hidden_state[:, 0, :]
                
            encoder_outputs = encoder_outputs.view(batch_size, max_num_segments, self.hidden_size)
            attention_mask = attention_mask.any(dim=2)
            
        ## encode each example by aggregating Bert segment outputs
        outputs, hidden = self.segment_encoder(inputs=encoder_outputs, attention_mask=attention_mask)
        return outputs, hidden

class GraphDistillationLayer(MessagePassing):
    def __init__(self, hidden_dim, drop=0.5):
        super().__init__(aggr='add')
        
        self.lin1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.empty(hidden_dim))
        
        self.dropout = nn.Dropout(drop)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bias.data.zero_()
        
    def forward(self, x, edge_index):
        row, col = edge_index # [num edges, num edges]
        deg = degree(col, x.size(0), dtype=x.dtype).to(x.device) # [num nodes,]
        deg_inv = deg.pow(-1)
        norm = deg_inv[row] # [num edges,]
        
        out = self.dropout(self.lin1(x)) 
        
        out -= self.propagate(edge_index, x=x, norm=norm)
        out += self.bias
        
        return out
        
    def message(self, x_i, x_j, norm): # [num edges, hidden dim], [num edges, hidden dim], [num edges,]
        return norm.unsqueeze(1) * self.dropout(self.lin2(torch.cat([x_i, x_j], dim=-1))) # [num edges, hidden dim]
    
class GraphDistillationNetwork(nn.Module):
    def __init__(self, num_gdl, hidden_dim):
        super().__init__()
        
        self.num_gdl = num_gdl
        self.gdl = nn.ModuleList([GraphDistillationLayer(hidden_dim) for i in range(num_gdl)])
        self.gelu = nn.GELU()
        
    def forward(self, x, edge_index):
        for k in range(self.num_gdl):
            x = self.gdl[k](x, edge_index)
            x = self.gelu(x)
        return x
        

class LADANForTextClassificationOutput(ModelOutput):
    loss = None
    preds = None
    community_preds = None       

class LADANForTextClassification(nn.Module):
    def __init__(self, hier_encoder, graph_distillation, num_communities, num_labels, community_weights=None, label_weights=None, drop=0.5):
        super().__init__()
        
        self.hidden_size = 768 #hier_encoder.hidden_size
        self.num_communities = num_communities
        self.num_labels = num_labels
        self.hier_encoder = hier_encoder
        self.graph_distillation = graph_distillation
        self.label_context_tf = nn.Linear(2*768, 768)
        self.final_tf = LstmAttn(768)
        self.community_fc = nn.Linear(768, num_communities)
        self.classifier_fc = nn.Linear(768, num_labels)
        
        if community_weights is None:
            community_weights = torch.ones(num_communities)
        self.community_loss_fct = nn.BCEWithLogitsLoss(community_weights)
        
        if label_weights is None:
            label_weights = torch.ones(num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss(label_weights)
        
        self.community_loss_factor = 0.5
        
        self.dropout = nn.Dropout(drop)
        self.counter = 0
        
    def forward(self,
                input_ids=None, # [batch size, max segments, max segment size]
                attention_mask=None, # [batch size, max segments, max segment size]
                label_input_ids=None, # [num labels, max label segments, max label segment size]
                label_attention_mask=None, # [num labels, max label segments, max label segment size]
                community2label=None, # [num comms, num labels]
                edges=None, # [2, num edges]
                community_labels=None, # [batch size, num comm labels]
                labels=None): # [batch size, num labels]
        
        self.counter += 1
        
        torch.cuda.empty_cache()
        input_hidden_states, input_basic_states = self.hier_encoder(input_ids=input_ids, attention_mask=attention_mask, segment_size=None) # [batch size, max segments, hidden dim], [batch size, hidden dim]
        
        community_logits = self.dropout(self.community_fc(input_basic_states)) # [batch size, num comm labels]
        community_preds = (torch.sigmoid(community_logits) > 0.5).float() # [batch size, num comm labels]
        
        input_final_states = input_basic_states.clone()
        
        try:
            label_hidden_states = self.hier_encoder(input_ids=label_input_ids, attention_mask=label_attention_mask, segment_size=None)[1] # [num labels, hidden dim]
            label_distinguish_states = self.graph_distillation(label_hidden_states, edges) # [num labels, hidden dim]        
                
            for comm_idx in range(community_preds.size(1)):
                community_flag = community_labels[:, comm_idx] if self.training else community_preds[:, comm_idx] # [batch size,]
                if community_flag.sum() == 0:
                    continue
                label_flag = community2label[comm_idx, :] # [num labels,]
                
                valid_label_distinguish_states = label_distinguish_states[label_flag.bool(), :] # [num valid labels, hidden dim]
                
                label_context = torch.cat([valid_label_distinguish_states.max(dim=0)[0], valid_label_distinguish_states.min(dim=0)[0]]) # [2 * hidden dim,]
                
                label_context = self.label_context_tf(label_context) # [hidden dim]
                
                input_distinguish_states = self.final_tf(inputs=input_hidden_states[community_flag.bool(), :, :], attention_mask=attention_mask.any(dim=-1)[community_flag.bool(), :], dynamic_context=label_context)[1] # [num pos comm idx, hidden dim]
                
                input_final_states[community_flag.bool(), :] += input_distinguish_states 
            
        except torch.cuda.OutOfMemoryError:
            print("+++ Skipping fusion")
            
        logits = self.dropout(self.classifier_fc(input_final_states)) # [batch size, num labels]
        preds = (torch.sigmoid(logits) > 0.5).float() # [batch size, num labels]
        
        # print(community_logits.shape, community_labels.shape, logits.shape, labels.shape)
        community_loss = self.community_loss_fct(community_logits, community_labels)
        non_community_loss = self.loss_fct(logits, labels)
        loss = self.community_loss_factor * community_loss + non_community_loss
        
        return LADANForTextClassificationOutput(loss=loss, preds=preds, community_preds=community_preds)
    

bert = AutoModel.from_pretrained(model_src) #, cache_dir='~/HDD/LSI-Cache')
hier_bert = HierBert(bert)
hier_bert.gradient_checkpointing_enable()
graph_distillation = GraphDistillationNetwork(2, 768)
model = LADANForTextClassification(hier_bert, graph_distillation, num_communities, len(label_vocab), community_weights, label_weights).to(DEVICE)

# Compute macro F1 scores
def compute_metrics(p, threshold=0.5):
    metrics = {}
    
    preds = (p.predictions[0] > threshold).astype(float)
    refs = p.label_ids[0]
    metrics['prec'] = precision_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['rec'] = recall_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['f1'] = f1_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    print(preds)
    print(refs)
    community_preds = (p.predictions[1] > threshold).astype(float)
    community_refs = p.label_ids[1]
    print(community_preds)
    print(community_refs)
    metrics['c-prec'] = precision_score(community_refs, community_preds, average='macro', labels=list(range(len(community2label))))
    metrics['c-rec'] = recall_score(community_refs, community_preds, average='macro', labels=list(range(len(community2label))))
    metrics['c-f1'] = f1_score(community_refs, community_preds, average='macro', labels=list(range(len(community2label))))
    return metrics

# Different layers have different learning rates, here we set the learning rates for each layer
def AdamWLLRD(model, bert_lr=3e-5, intermediate_lr=5e-5, top_lr=1e-3, wd=1e-2):
    opt_params = []
    named_params = list(model.named_parameters())
    print([n for (n, p) in named_params])

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert = ["hier_encoder.bert_encoder"]    
    intermediate = ["hier_encoder.segment_encoder"]
    top = ["graph_distillation", "final_tf", "label_context_tf", "classifier_fc", "community_fc"]

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

opt = AdamWLLRD(model)
sch = transformers.get_linear_schedule_with_warmup(opt, num_warmup_steps=90, num_training_steps=10000)

training_args = TrainingArguments(
    output_dir=output_fol,
    overwrite_output_dir=False,
    do_train=False,
    do_eval=False,
    do_predict=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=24,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=3,
    eval_accumulation_steps=None,
    learning_rate=None,
    weight_decay=None,
    num_train_epochs=15,
    logging_strategy='steps',
    logging_steps=100,
    logging_first_step=False,
    save_strategy='epoch',
    save_total_limit=1,
    seed=42,
    fp16=True,
    dataloader_num_workers=8,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    group_by_length=False,
    dataloader_pin_memory=True,
    resume_from_checkpoint=False,
    gradient_checkpointing=False,
    label_names=['labels', 'community_labels']
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset['train'],
    eval_dataset=dataset['dev'],
    compute_metrics=compute_metrics,
    optimizers=(opt, sch)
)

if training_args.do_train:
    _, _, metrics = trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
    trainer.save_metrics('train', metrics)
    
if training_args.do_eval:
    test_results = trainer.evaluate(eval_dataset=dataset['test'])
    print(test_results)
    trainer.save_metrics('test', test_results)
    
if training_args.do_predict:
    predictions, label_ids, results = trainer.predict(test_dataset=dataset['test'])
    np.save(os.path.join(output_fol, "predictions.npy"), predictions)
    np.save(os.path.join(output_fol, "label_ids.npy"), label_ids)
    print(results)
    