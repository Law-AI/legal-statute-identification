
import datasets
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
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import BatchEncoding, TrainingArguments, Trainer, AdamW
from transformers.file_utils import ModelOutput
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

root = sys.argv[1]                  # Dataset directory
model_src = sys.argv[2]             # Model name/directory
output_fol = sys.argv[3]            # Output Folder
CACHE_DIR = "~/HDD/LSI-Cache"

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

label_schema = Features(
    {
        "id": Value('string'),
        "title": Value('string'),
        "text": Sequence(Value('string'))
    }
)


dataset = load_dataset('json', data_files={'train': os.path.join(root, "train.json"), 
                                           'dev': os.path.join(root, "dev.json"), 
                                           'test': os.path.join(root, "test-expln-noimp.json")}, 
                       field='data', 
                       cache_dir=CACHE_DIR)

label_dataset = load_dataset('json', data_files={'label': os.path.join(root, "label_descriptions.json")}, 
                             field='data', 
                             cache_dir='~/HDD/LSI-Cache')

dataset = dataset.map(schema.encode_example, features=schema)
dataset = dataset.filter(lambda example: len(example['text']) != 0)
label_dataset = label_dataset.map(label_schema.encode_example, features=label_schema)

config = AutoConfig.from_pretrained(model_src) #, cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(model_src)
special_tokens = {'additional_special_tokens': ['<ENTITY>', '<ACT>', '<SECTION>']}
tokenizer.add_special_tokens(special_tokens)
tokenizer.add_tokens(special_tokens['additional_special_tokens'])

assert tokenizer.pad_token_id is not None
assert tokenizer.bos_token_id is not None
assert tokenizer.eos_token_id is not None

dataset = dataset.map(lambda example: tokenizer(list(example['text']), return_token_type_ids=False), 
                      batched=False)
label_dataset = label_dataset.map(lambda example: tokenizer([example['title'] + ": "] + list(example['text']),
                                                            return_token_type_ids=False), 
                                  batched=False)

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
label_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

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
    label_weights = generate_label_weights().cuda()
    with open(os.path.join(root, 'label_weights.pkl'), 'wb') as fw:
        pkl.dump(label_weights, fw)

# Data collator
def collate_fn(examples):
    max_segment_size = min(max(sum(s.size(0) - 2 for s in exp['input_ids']) for exp in examples), 4096)
    haslabels = True if 'labels' in examples[0] else False
    input_ids = torch.zeros(len(examples), max_segment_size, dtype=torch.long).fill_(tokenizer.pad_token_id)
    if haslabels:
        labels = torch.zeros(len(examples), len(label_vocab))
    for exp_idx, exp in enumerate(examples):
        exp_text = torch.cat([s[1:-1] for s in exp['input_ids']], dim=0)[:max_segment_size - 2] if len(exp['input_ids']) > 1 else exp['input_ids'][0][:max_segment_size - 2]
        input_ids[exp_idx, 1:len(exp_text)+1] = exp_text
        input_ids[exp_idx, 0] = tokenizer.bos_token_id
        input_ids[exp_idx, len(exp_text)] = tokenizer.eos_token_id
        if haslabels:
            labels[exp_idx].scatter_(0, exp['labels'], 1.)
        
    attention_mask = input_ids != tokenizer.pad_token_id
    global_attention_mask = input_ids == tokenizer.bos_token_id
    
    if haslabels:
        return BatchEncoding({'input_ids': input_ids, 
                              'attention_mask': attention_mask, 
                              'global_attention_mask': global_attention_mask,
                              'labels': labels, 
                              'label_input_ids': label_batch.input_ids, 
                              'label_attention_mask': label_batch.attention_mask,
                              'label_global_attention_mask': label_batch.global_attention_mask})
    else:
        return BatchEncoding({'input_ids': input_ids, 
                              'attention_mask': attention_mask,
                              'global_attention_mask': global_attention_mask})


label_loader = torch.utils.data.DataLoader(label_dataset['label'], 
                                           batch_size=len(label_vocab), 
                                           collate_fn=collate_fn)
for label_batch in label_loader:
    pass
print(label_batch.input_ids.shape)
# Main BAF layer
class BilinearAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, drop=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        self.input_transform = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.label_transform = nn.Linear(self.hidden_dim, self.intermediate_dim)
        
        self.input_attention = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.label_attention = nn.Linear(self.hidden_dim, self.intermediate_dim)
        
        self.pool = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
        self.context = nn.Parameter(torch.rand(self.intermediate_dim))
        
        self.dropout = nn.Dropout(drop)
        
    def forward(self, 
                input_hidden_states=None, # [batch size, max seq len, hidden dim]
                label_hidden_states=None): # [num labels, hidden dim]
        
        batch_size, max_seq_len = input_hidden_states.shape[:2]
        num_labels = label_hidden_states.size(0)
        
        label_hidden_states = label_hidden_states.unsqueeze(0).repeat(batch_size, 1, 1) # [batch size, num labels, intermediate dim] 
        
        input_trans_states = torch.sigmoid(self.dropout(self.input_transform(input_hidden_states))) # [batch size, max seq len, intermediate dim]
        label_trans_states = torch.sigmoid(self.dropout(self.label_transform(label_hidden_states))) # [batch size, num labels, intermediate dim] 

        input_attention_states = torch.sigmoid(self.dropout(self.input_attention(input_hidden_states))) # [batch size, max seq len, intermediate dim]
        label_attention_states = torch.sigmoid(self.dropout(self.label_attention(label_hidden_states))) # [batch size, num labels, intermediate dim]1794

        attention_matrix = torch.outer(torch.ones(max_seq_len, device=input_hidden_states.device), self.context) # [max seq len, intermediate dim]
        
        attention_matrix = torch.bmm(attention_matrix.unsqueeze(0) * input_attention_states, label_attention_states.permute(0, 2, 1)) # [batch size, max seq len, num labels]
        attention_matrix = F.softmax(attention_matrix, dim=-1)
        
        input_trans_states = input_trans_states.permute(0, 2, 1).unsqueeze(2).reshape(-1, 1, max_seq_len) # [batch size * intermediate dim, 1, max seq len]
        attention_matrix = attention_matrix.unsqueeze(1).repeat(1, self.intermediate_dim, 1, 1).view(-1, max_seq_len, num_labels) # [batch size * intermediate dim, max seq len, num labels]
        label_trans_states = label_trans_states.permute(0, 2, 1).unsqueeze(3).reshape(-1, num_labels, 1) # [batch size * intermediate dim, num labels, 1]
        input_fusion_states = torch.bmm(input_trans_states, torch.bmm(attention_matrix, label_trans_states)).squeeze().view(batch_size, self.intermediate_dim)  # [batch size, intermediate dim]
        
        input_fusion_states = self.dropout(self.pool(input_fusion_states)) # [batch size, hidden dim]
         
        return input_fusion_states       
        
        

@dataclass
class TextClassifierOutput(ModelOutput):
    loss:torch.Tensor = None
    logits:torch.Tensor = None
    hidden_states:torch.Tensor = None

# BAF + Longformer Encoder
class BertBAFForTextClassfication(nn.Module):
    def __init__(self, encoder, baf, num_labels, label_weights=None, segment_size=None, drop=0.5):
        super().__init__()
        
        self.hidden_size = encoder.config.hidden_size
        self.num_labels = num_labels
        self.encoder = encoder
        self.baf = baf
        self.classifier_fc = nn.Linear(encoder.config.hidden_size, num_labels)
        
        if label_weights is None:
            label_weights = torch.ones(num_labels)
            
        self.loss_fct = nn.BCEWithLogitsLoss(label_weights)
        
        self.segment_size = segment_size
        
        self.dropout = nn.Dropout(drop)
    
    def gradient_checkpointing_enable(self):
        self.hier_encoder.gradient_checkpointing_enable()

    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                global_attention_mask=None,
                labels=None,
                label_input_ids=None,
                label_attention_mask=None,
                label_global_attention_mask=None
                ):
        
        input_hidden_states = self.dropout(self.encoder(input_ids=input_ids, 
                                                        attention_mask=attention_mask,
                                                        global_attention_mask=global_attention_mask).last_hidden_state)
        label_input_ids_short = label_input_ids
        label_attention_mask_short = label_attention_mask
        label_global_attention_mask_short = label_global_attention_mask
          
        try:
            if self.segment_size is None:
                label_hidden_states = self.dropout(self.encoder(input_ids=label_input_ids_short, 
                                                        attention_mask=label_attention_mask_short,
                                                        global_attention_mask=label_global_attention_mask_short).last_hidden_state[:, 0, :])
            else:
                label_hidden_states = []
                for i in range(0, label_input_ids.size(0), self.segment_size):
                    label_hidden_states.append(self.dropout(self.encoder(input_ids=label_input_ids_short[i:i+self.segment_size, :],
                                                                            attention_mask=label_attention_mask_short[i:i+self.segment_size, :],
                                                                            global_attention_mask=label_global_attention_mask_short[i:i+self.segment_size, :]).last_hidden_state[:, 0, :]))
                label_hidden_states = torch.cat(label_hidden_states, dim=0)
                
            input_fusion_states = self.baf(input_hidden_states, label_hidden_states)
        
        except torch.cuda.OutOfMemoryError: # Skips the fusion step if GPU is out of memory instead of exiting
            print("+++ Skipping fusion")
            input_fusion_states = None
        
        input_final_states = input_hidden_states[:, 0, :] + input_fusion_states if input_fusion_states is not None else input_hidden_states[:, 0, :]
        
        logits = self.dropout(self.classifier_fc(input_final_states))
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        
        return TextClassifierOutput(loss=loss, logits=torch.sigmoid(logits), hidden_states=input_final_states) 


bert = AutoModel.from_pretrained(model_src) #, cache_dir=CACHE_DIR)
bert.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
baf = BilinearAttentionFusion(768, 512)
model = BertBAFForTextClassfication(bert, baf, len(label_vocab), label_weights=label_weights, segment_size=None).cuda()

# Compute macro F1 scores
def compute_metrics(p, threshold=0.6):
    metrics = {}
    preds = (p.predictions > threshold).astype(float)
    refs = p.label_ids
    metrics['prec'] = precision_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['rec'] = recall_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['f1'] = f1_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    return metrics

# Different layers have different learning rates, here we set the learning rates for each layer
def AdamWLLRD(model, bert_lr=5e-5, intermediate_lr=1e-3, top_lr=1e-3, wd=1e-2):
    opt_params = []
    named_params = list(model.named_parameters())
    
    print(named_params)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert = ["encoder.embeddings", "encoder.encoder", "encoder.pooler"]    
    intermediate = []
    top = ["baf", "classifier_fc"]

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
sch = transformers.get_linear_schedule_with_warmup(opt, num_training_steps=345, 
                                                   num_warmup_steps=3600)

training_args = TrainingArguments(
    output_dir=output_fol,
    overwrite_output_dir=False,
    do_train=False,
    do_eval=False,
    do_predict=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    eval_accumulation_steps=None,
    learning_rate=None,
    weight_decay=None,
    num_train_epochs=25,
    logging_strategy='epoch',
    logging_first_step=False,
    save_strategy='epoch',
    save_total_limit=3,
    seed=42,
    fp16=True,
    dataloader_num_workers=1,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    group_by_length=False,
    dataloader_pin_memory=True,
    resume_from_checkpoint=False,
    gradient_checkpointing=False,
    label_names=['labels']
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
    _, _, metrics = trainer.train(ignore_keys_for_eval=['hidden_states'], resume_from_checkpoint=False)
    trainer.save_model()
    trainer.save_metrics('train', metrics)
if training_args.do_eval:
    model.load_state_dict(torch.load(os.path.join(output_fol, "pytorch_model.bin"), map_location='cuda'))
    test_results = trainer.evaluate(eval_dataset=dataset['test'], ignore_keys=['hidden_states'])
    print(test_results)
    trainer.save_metrics('test', test_results)
if training_args.do_predict:
    model.load_state_dict(torch.load(os.path.join(output_fol, "pytorch_model.bin"), map_location='cuda'))
    predictions, label_ids, results = trainer.predict(test_dataset=dataset['test'], ignore_keys=['hidden_states'])
    np.save(os.path.join(output_fol, "predictions.npy"), predictions)
    np.save(os.path.join(output_fol, "label_ids.npy"), label_ids)
    print(results)