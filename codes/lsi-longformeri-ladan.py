# %%
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
#from hibert_model import *
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

# torch.autograd.set_detect_anomaly(True)

# %%
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
# In[17]:

root = sys.argv[1]
model_src = sys.argv[2]
output_fol = sys.argv[3]

with open(os.path.join(root, "label_vocab.json")) as fr:
    label_vocab = json.load(fr)

with open(os.path.join(root, "label2community_tfidf.json")) as fr:
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

# %%
dataset = load_dataset('json', data_files={'train': os.path.join(root, "train2.json"), 
                                           'dev': os.path.join(root, "dev2.json"), 
                                           'test': os.path.join(root, "test2-expln3-noimp.json")}, field='data', 
                       cache_dir='Cache')
label_dataset = load_dataset('json', data_files={'label': os.path.join(root, "label_descriptions.json")}, field='data', cache_dir='Cache')


# %%
dataset = dataset.map(schema.encode_example, features=schema)
dataset = dataset.filter(lambda example: len(example['text']) != 0)
label_dataset = label_dataset.map(label_schema.encode_example, features=label_schema)

dataset['train'] = dataset['train'].select([0])
dataset['dev'] = dataset['dev'].select([0])

# %%
config = AutoConfig.from_pretrained(model_src)
tokenizer = AutoTokenizer.from_pretrained(model_src) #, cache_dir='~/HDD/LSI-Cache')
special_tokens = {'additional_special_tokens': ['<ENTITY>', '<ACT>', '<SECTION>']}
tokenizer.add_special_tokens(special_tokens)
tokenizer.add_tokens(special_tokens['additional_special_tokens'])

dataset = dataset.map(lambda example: tokenizer(list(example['text']), return_token_type_ids=False), batched=False)
label_dataset = label_dataset.map(lambda example: tokenizer([example['title'] + ": "] + list(example['text']), return_token_type_ids=False), batched=False)

# %%
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
label_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# %%
def generate_community_weights():
    weights = torch.zeros(num_communities)
    for exp in tqdm(dataset['train'], desc="Label Weights"):
        _weights = torch.zeros(num_communities)
        for l in exp['labels']:
            _weights[label2community[l.item()]] = 1
        weights += _weights
    return len(dataset['train']) / weights

if os.path.exists(os.path.join(root, 'community_weights_tfidf.pkl')):
    with open(os.path.join(root, 'community_weights_tfidf.pkl'), 'rb') as fr:
        community_weights = pkl.load(fr)
else:
    community_weights = generate_community_weights() #.to(DEVICE)
    with open(os.path.join(root, 'community_weights_tfidf.pkl'), 'wb') as fw:
        pkl.dump(community_weights, fw)

# %%
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
    
with open(os.path.join(root, 'ladan_edges_tfidf.json')) as fr:
    edges = json.load(fr)
edges = torch.tensor(edges).permute(1, 0) #.to(DEVICE) # [2, num edges]

# %%
def collate_fn(examples):
    
    max_segment_size = min(max(sum(s.size(0) - 2 for s in exp['input_ids']) for exp in examples), 4096)
    haslabels = True if 'labels' in examples[0] else False
    input_ids = torch.zeros(len(examples), max_segment_size, dtype=torch.long).fill_(tokenizer.pad_token_id)
    
    if haslabels:
        labels = torch.zeros(len(examples), len(label_vocab))
        community_labels = torch.zeros(len(examples), num_communities)
        
    for exp_idx, exp in enumerate(examples):
        # print(exp['input_ids'])
        exp_text = torch.cat([s[1:-1] for s in exp['input_ids']], dim=0)[:max_segment_size - 2] if len(exp['input_ids']) > 1 else exp['input_ids'][0][:max_segment_size - 2]
        
        input_ids[exp_idx, 1:len(exp_text)+1] = exp_text
        input_ids[exp_idx, 0] = tokenizer.bos_token_id
        input_ids[exp_idx, len(exp_text)] = tokenizer.eos_token_id
        
        if haslabels:
            labels[exp_idx].scatter_(0, exp['labels'], 1.)
            comms = torch.tensor(list(set(label2community[l.item()] for l in exp['labels'])), dtype=torch.long)
            community_labels[exp_idx].scatter_(0, comms, 1.)
  
    attention_mask = input_ids != tokenizer.pad_token_id
    global_attention_mask = input_ids == tokenizer.bos_token_id
    
    
    if haslabels:
        # print(input_ids.shape, attention_mask.shape, label_batch.input_ids.shape, label_batch.attention_mask.shape, community2label.shape, edges.shape, labels.shape, community_labels.shape)
        return BatchEncoding({'input_ids': input_ids, 
                              'attention_mask': attention_mask, 
                              'global_attention_mask': global_attention_mask,
                              'label_input_ids': label_batch.input_ids, 
                              'label_attention_mask': label_batch.attention_mask,
                              'label_global_attention_mask': label_batch.global_attention_mask, 
                              'community2label': community2label, 
                              'edges': edges, 
                              'labels': labels, 
                              'community_labels': community_labels})
    else:
        return BatchEncoding({'input_ids': input_ids, 
                              'attention_mask': attention_mask,
                              'global_attention_mask': global_attention_mask})

# %%
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
        # inputs_packed = pack_padded_sequence(inputs, torch.clamp(lengths, min=1).cpu(), enforce_sorted=False, batch_first=True)  
        # outputs_packed = self.lstm(inputs_packed)[0]
        # outputs = pad_packed_sequence(outputs_packed, batch_first=True)[0]
        
        outputs = self.lstm(inputs)[0]  
        
        activated_outputs = torch.tanh(self.dropout(self.attn_fc(outputs)))
        context = dynamic_context if dynamic_context is not None else self.context
        context = context.expand(inputs.size(0), self.hidden_size)
        scores = torch.bmm(activated_outputs, context.unsqueeze(2)).squeeze(2)
        
        # print(inputs.shape, outputs.shape, attention_mask.shape, scores.shape)
        
        masked_scores = scores.masked_fill(~attention_mask, -1e-32)
        masked_scores = F.softmax(masked_scores, dim=1)
        
        hidden = torch.sum(outputs * masked_scores.unsqueeze(2), dim=1)
        return outputs, hidden
    

class LongformerInternal(nn.Module):
    def __init__(self, encoder, drop=0.5):
        super().__init__()
        
        self.bert_encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.segment_encoder = LstmAttn(self.hidden_size, drop=drop)
        self.dropout = nn.Dropout(drop)
   
    def gradient_checkpointing_enable(self):
        self.bert_encoder.gradient_checkpointing_enable()

    def _encoder_forward(self, input_ids, attention_mask, global_attention_mask, dummy):
        # print(self.bert_encoder(input_ids=input_ids, 
        #                                       attention_mask=attention_mask).pooler_output.shape)
        # print(self.bert_encoder(input_ids=input_ids, 
        #                                       attention_mask=attention_mask).last_hidden_state.shape)
        intermediate = self.bert_encoder(input_ids=input_ids, 
                                        attention_mask=attention_mask,
                                        global_attention_mask=global_attention_mask)
        outputs = self.dropout(intermediate.last_hidden_state)
        hidden = self.dropout(intermediate.last_hidden_state)[:, 0, :]
        return outputs, hidden
    
    def forward(self, input_ids=None, 
                attention_mask=None, 
                global_attention_mask = None, 
                encoder_outputs=None, 
                dynamic_context=None):
        if input_ids is not None:
            batch_size, max_seq_len = input_ids.shape
        
        ## encode individual segments using Bert
        if input_ids is not None:
            dummy = torch.ones(1, dtype=torch.float, requires_grad=True)
            # outputs, hidden = checkpoint(self._encoder_forward, input_ids, attention_mask, global_attention_mask, dummy)
            outputs, hidden = self._encoder_forward(input_ids, attention_mask, global_attention_mask, dummy)
            # print(encoder_outputs.shape)
            # encoder_outputs = encoder_outputs.view(batch_size, self.hidden_size)
            # print(encoder_outputs.shape)
            # attention_mask = attention_mask.any(dim=-1)
            
        ## encode each example by aggregating Bert segment outputs
        # outputs, hidden = self.segment_encoder(inputs=encoder_outputs, 
        #                                        attention_mask=attention_mask, 
        #                                        dynamic_context=dynamic_context)
        #outputs, hidden = encoder_outputs, None
        # (batch, seq, emb) AND (batch, emb)
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
        # TODO
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
                input_ids=None, # [batch size, max segment size]
                attention_mask=None, # [batch size, max segment size]
                global_attention_mask=None, # same as above
                label_input_ids=None, # [num labels, max label segment size]
                label_attention_mask=None, # [num labels, max label segment size]
                label_global_attention_mask=None, # same as above
                community2label=None, # [num comms, num labels]
                edges=None, # [2, num edges]
                community_labels=None, # [batch size, num comm labels]
                labels=None): # [batch size, num labels]
        
        # print("Model:", community_labels.shape)
        # print("Count", self.counter)
        self.counter += 1
        
        torch.cuda.empty_cache()
        input_hidden_states, input_basic_states = self.hier_encoder(input_ids=input_ids, 
                                                                    attention_mask=attention_mask, 
                                                                    global_attention_mask=global_attention_mask) 
        # [batch size, max segments, hidden dim], [batch size, hidden dim]
        
        community_logits = self.dropout(self.community_fc(input_basic_states)) # [batch size, num comm labels]
        community_preds = (torch.sigmoid(community_logits) > 0.25).float() # [batch size, num comm labels]
        
        # print("Model:", community_logits.shape, community_labels.shape)
        
        input_final_states = input_basic_states.clone()
        
        try:
            label_hidden_states = self.hier_encoder(input_ids=label_input_ids, 
                                                    attention_mask=label_attention_mask, 
                                                    global_attention_mask=label_global_attention_mask)[1] # [num labels, hidden dim]
            
            label_distinguish_states = self.graph_distillation(label_hidden_states, edges) # [num labels, hidden dim]     
            # label_distinguish_states = label_hidden_states   
            # print('ShittyCode')
            # print(label_hidden_states, label_distinguish_states)
                
            for comm_idx in range(community_preds.size(1)):
                community_flag = community_labels[:, comm_idx] if self.training else community_preds[:, comm_idx] # [batch size,]
                if community_flag.sum() == 0:
                    continue
                label_flag = community2label[comm_idx, :] # [num labels,]
                
                # valid --> no. of labels in current comm
                # valid_label_hidden_states = self.hier_encoder(input_ids=label_input_ids[label_flag.bool(), :, :], attention_mask=label_attention_mask[label_flag.bool(), :, :], segment_size=None)[1] # [num valid labels, hidden dim]
                # valid_label_distinguish_states = self.graph_distillation(valid_label_hidden_states, edges) # [num valid labels, hidden dim]
                
                
                
                valid_label_distinguish_states = label_distinguish_states[label_flag.bool(), :] # [num valid labels, hidden dim]
                
                label_context = torch.cat([valid_label_distinguish_states.max(dim=0)[0], valid_label_distinguish_states.min(dim=0)[0]]) # [2 * hidden dim,]
                
                label_context = self.label_context_tf(label_context) # [hidden dim]
                
                input_distinguish_states = self.final_tf(inputs=input_hidden_states[community_flag.bool(), :, :],
                                                         attention_mask=attention_mask[community_flag.bool(), :],
                                                         dynamic_context=label_context)[1] # [num pos comm idx, hidden dim]
                
                input_final_states[community_flag.bool(), :] += input_distinguish_states 
            
        except torch.cuda.OutOfMemoryError:
            print("+++ Skipping fusion")
            
        logits = self.dropout(self.classifier_fc(input_final_states)) # [batch size, num labels]
        preds = (torch.sigmoid(logits) > 0.5).float() # [batch size, num labels]
        
        # print(community_logits.shape, community_labels.shape, logits.shape, labels.shape)
        community_loss = self.community_loss_fct(community_logits, community_labels)
        non_community_loss = self.loss_fct(logits, labels)
        loss = self.community_loss_factor * community_loss + non_community_loss
        # print('Loss')
        # print(loss)
        # print(loss, community_loss, non_community_loss)
        # print(logits, labels)
        
        return LADANForTextClassificationOutput(loss=loss, preds=preds, community_preds=community_preds)
    

# %%
# hier_bert = StasModel.from_pretrained(model_src, cache_dir='~/HDD/LSI-Cache')
bert = AutoModel.from_pretrained(model_src) #, cache_dir='~/HDD/LSI-Cache')
bert.resize_token_embeddings(len(tokenizer), 
                                 pad_to_multiple_of=8)

hier_bert = LongformerInternal(bert)
hier_bert.gradient_checkpointing_enable()
graph_distillation = GraphDistillationNetwork(2, 768)
model = LADANForTextClassification(hier_bert, graph_distillation, num_communities, len(label_vocab), community_weights, label_weights).to(DEVICE)

# %%
def compute_metrics(p, threshold=0.5):
    metrics = {}
    
    preds = (p.predictions[0] > threshold).astype(float)
    refs = p.label_ids[0]
    metrics['prec'] = precision_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['rec'] = recall_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['f1'] = f1_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    # print(preds)
    # print(refs)
    community_preds = (p.predictions[1] > threshold).astype(float)
    community_refs = p.label_ids[1]
    # print(community_preds)
    # print(community_refs)
    metrics['c-prec'] = precision_score(community_refs, community_preds, average='macro', labels=list(range(len(community2label))))
    metrics['c-rec'] = recall_score(community_refs, community_preds, average='macro', labels=list(range(len(community2label))))
    metrics['c-f1'] = f1_score(community_refs, community_preds, average='macro', labels=list(range(len(community2label))))
    return metrics

# %%
def AdamWLLRD(model, bert_lr=1e-4, intermediate_lr=5e-4, top_lr=1e-3, wd=1e-2):
    opt_params = []
    named_params = list(model.named_parameters())
    # print([n for (n, p) in named_params])

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
sch = transformers.get_linear_schedule_with_warmup(opt, num_warmup_steps=1670, num_training_steps=18000)

training_args = TrainingArguments(
    output_dir=output_fol,
    overwrite_output_dir=False,
    do_train=False,
    do_eval=False,
    do_predict=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    eval_accumulation_steps=None,
    learning_rate=None,
    weight_decay=None,
    num_train_epochs=25,
    logging_strategy='steps',
    logging_steps=100,
    logging_first_step=False,
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
    resume_from_checkpoint=False,
    gradient_checkpointing=False,
    label_names=['labels', 'community_labels']
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset['train'],#.select(list(range(2000))),
    eval_dataset=dataset['dev'],#.select(list(range(200))),
    compute_metrics=compute_metrics,
    optimizers=(opt, sch)
)

if training_args.do_train:
    _, _, metrics = trainer.train(resume_from_checkpoint=False)
    #torch.save(model.state_dict(), os.path.join(root, output_fol, "pytorch_model.bin"))
    trainer.save_model()
    trainer.save_metrics('train', metrics)
    
if training_args.do_eval:
    #dev_results = trainer.evaluate(ignore_keys=['hidden_states'])
    model.load_state_dict(torch.load(os.path.join(output_fol, "pytorch_model.bin"), map_location=DEVICE))
    test_results = trainer.evaluate(eval_dataset=dataset['test'])
    #print(dev_results)
    print(test_results)
    trainer.save_metrics('test', test_results)
    #with open(os.path.join(root, output_fol, "eval_results.json"), 'w') as fw:
    #json.dump(test_results, fw, indent=4)
    
if training_args.do_predict:
    model.load_state_dict(torch.load(os.path.join(output_fol, "pytorch_model.bin"), map_location=DEVICE))
    predictions, label_ids, results = trainer.predict(test_dataset=dataset['test'])
    
    np.save(os.path.join(output_fol, "predictions_expln3_noimp.npy"), predictions[0])
    np.save(os.path.join(output_fol, "label_ids_expln3_noimp.npy"), label_ids[0])
    np.save(os.path.join(output_fol, "predictions_comm_expln3_noimp.npy"), predictions[1])
    np.save(os.path.join(output_fol, "label_ids_comm_expln3_noimp.npy"), label_ids[1])
    trainer.save_metrics('test2_expln3_noimp', results)
    print(results)
