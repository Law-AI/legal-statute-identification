
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle as pkl
import sys
import json
import transformers
import time


from datasets import Features, Sequence, load_dataset
from datasets.features import ClassLabel, Value
from stas.model_fast import *

from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding, TrainingArguments, Trainer, AdamW
from transformers.file_utils import ModelOutput
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


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

label_schema = Features(
    {
        "id": Value('string'),
        "title": Value('string'),
        "text": Sequence(Value('string'))
    }
)


dataset = load_dataset('json', data_files={'train': os.path.join(root, "train2.json"), 
                                           'dev': os.path.join(root, "dev2.json"), 
                                           'test': os.path.join(root, "test2-expln3-noimp.json")}, field='data', 
                       cache_dir='~/HDD/LSI-Cache')
label_dataset = load_dataset('json', data_files={'label': os.path.join(root, "label_descriptions.json")}, field='data', cache_dir='~/HDD/LSI-Cache')


dataset = dataset.map(schema.encode_example, features=schema)
label_dataset = label_dataset.map(label_schema.encode_example, features=label_schema)

tokenizer = AutoTokenizer.from_pretrained(model_src) #, cache_dir='~/HDD/LSI-Cache')
special_tokens = {'additional_special_tokens': ['<ENTITY>', '<ACT>', '<SECTION>']}
tokenizer.add_special_tokens(special_tokens)
tokenizer.add_tokens(special_tokens['additional_special_tokens'])

dataset = dataset.map(lambda example: tokenizer(list(example['text']), return_token_type_ids=False), batched=False)
label_dataset = label_dataset.map(lambda example: tokenizer([example['title'] + ": "] + list(example['text']), return_token_type_ids=False), batched=False)

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
        label_weights = pkl.load(fr).cuda()
else:
    label_weights = generate_label_weights().cuda()
    with open(os.path.join(root, 'label_weights.pkl'), 'wb') as fw:
        pkl.dump(label_weights, fw)

# Data collator
def collate_fn(examples):
    haslabels = True if 'labels' in examples[0] else False
    
    max_segments = min(max(len(exp['input_ids']) for exp in examples), 128)
    max_segment_size = min(max(max(len(sent) for sent in exp['input_ids']) for exp in examples), 128)
    
    input_ids = torch.zeros(len(examples), max_segments, max_segment_size, dtype=torch.long).fill_(tokenizer.pad_token_id)
    if haslabels:
        labels = torch.zeros(len(examples), len(label_vocab))
    
    for exp_idx, exp in enumerate(examples):
        for sent_idx, sent in enumerate(exp['input_ids'][:128]):
            sent = sent[:128]
            input_ids[exp_idx, sent_idx, :len(sent)] = sent
        if haslabels:
            labels[exp_idx].scatter_(0, exp['labels'], 1.)
        
    attention_mask = input_ids != tokenizer.pad_token_id
    
    if haslabels:
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'label_input_ids': label_batch.input_ids, 'label_attention_mask': label_batch.attention_mask})
    else:
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})


label_loader = torch.utils.data.DataLoader(label_dataset['label'], batch_size=len(label_vocab), collate_fn=collate_fn)
for label_batch in label_loader:
    pass
print(label_batch.input_ids.shape)

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
        label_attention_states = torch.sigmoid(self.dropout(self.label_attention(label_hidden_states))) # [batch size, num labels, intermediate dim]

        attention_matrix = torch.outer(torch.ones(max_seq_len, device=input_hidden_states.device), self.context) # [max seq len, intermediate dim]
        
        attention_matrix = torch.bmm(attention_matrix.unsqueeze(0) * input_attention_states, label_attention_states.permute(0, 2, 1)) # [batch size, max seq len, num labels]
        attention_matrix = F.softmax(attention_matrix, dim=-1)
        
        input_trans_states = input_trans_states.permute(0, 2, 1).unsqueeze(2).reshape(-1, 1, max_seq_len) # [batch size * intermediate dim, 1, max seq len]
        attention_matrix = attention_matrix.unsqueeze(1).repeat(1, self.intermediate_dim, 1, 1).view(-1, max_seq_len, num_labels) # [batch size * intermediate dim, max seq len, num labels]
        label_trans_states = label_trans_states.permute(0, 2, 1).unsqueeze(3).reshape(-1, num_labels, 1) # [batch size * intermediate dim, num labels, 1]
        input_fusion_states = torch.bmm(input_trans_states, torch.bmm(attention_matrix, label_trans_states)).squeeze().view(batch_size, self.intermediate_dim)  # [batch size, intermediate dim]
                
        input_fusion_states = self.dropout(self.pool(input_fusion_states)) # [batch size, hidden dim]
        return input_fusion_states       
        
        
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
        inputs_packed = pack_padded_sequence(inputs, torch.clamp(lengths, min=1).cpu(), enforce_sorted=False, batch_first=True)  
        outputs_packed = self.lstm(inputs_packed)[0]
        outputs = pad_packed_sequence(outputs_packed, batch_first=True)[0]
        
        activated_outputs = torch.tanh(self.dropout(self.attn_fc(outputs)))
        context = dynamic_context if dynamic_context is not None else self.context.expand(inputs.size(0), self.hidden_size)
        scores = torch.bmm(activated_outputs, context.unsqueeze(2)).squeeze(2)
        masked_scores = scores.masked_fill(~attention_mask, -1e-32)
        masked_scores = F.softmax(masked_scores, dim=1)
        
        hidden = torch.sum(outputs * masked_scores.unsqueeze(2), dim=1)
        return outputs, hidden


class TextClassifierOutput(ModelOutput):
    loss = None
    logits = None
    hidden_states = None


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
            
            valid_mask = attention_mask_flat.any(dim=-1)
            valid_input_ids_flat = input_ids_flat[valid_mask]
            valid_attention_mask_flat = attention_mask_flat[valid_mask]
            
            encoder_outputs = torch.zeros(batch_size * max_num_segments, self.hidden_size, device=input_ids.device)
                        
            if segment_size is not None:
                valid_encoder_outputs = []
                
                for fragment_idx, sent_idx in enumerate(range(0, valid_input_ids_flat.size(0), segment_size)):
                    print(fragment_idx)
                    input_ids_fragment = valid_input_ids_flat[sent_idx : sent_idx + segment_size]
                    attention_mask_fragment = valid_attention_mask_flat[sent_idx : sent_idx + segment_size]   
                    encoder_outputs_fragment = self.bert_encoder(input_ids_fragment, attention_mask_fragment).last_hidden_state[:, 0, :]
                    valid_encoder_outputs.append(encoder_outputs_fragment)
                
                valid_encoder_outputs = torch.cat(valid_encoder_outputs, dim=0)
                
            else:
                valid_encoder_outputs = self.bert_encoder(valid_input_ids_flat, valid_attention_mask_flat).last_hidden_state[:, 0, :]
            encoder_outputs[valid_mask] = valid_encoder_outputs
                
            encoder_outputs = encoder_outputs.view(batch_size, max_num_segments, self.hidden_size)
            attention_mask = attention_mask.any(dim=2)
            
        ## encode each example by aggregating Bert segment outputs
        
        outputs, hidden = self.segment_encoder(inputs=encoder_outputs, attention_mask=attention_mask)
        return outputs, hidden

# Complete BAF model
class BAFForTextClassfication(nn.Module):
    def __init__(self, hier_encoder, baf, num_labels, label_weights=None, drop=0.5):
        super().__init__()
        
        self.hidden_size = 768 #hier_encoder.hidden_size
        self.num_labels = num_labels
        self.hier_encoder = hier_encoder
        self.baf = baf
        self.classifier_fc = nn.Linear(768, num_labels)
        
        if label_weights is None:
            label_weights = torch.ones(num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss(label_weights)
        
        self.dropout = nn.Dropout(drop)
    
    def gradient_checkpointing_enable(self):
        self.hier_encoder.gradient_checkpointing_enable()

    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                labels=None,
                label_input_ids=None,
                label_attention_mask=None):
        
        torch.cuda.empty_cache()
        input_hidden_states = self.hier_encoder(input_ids=input_ids, attention_mask=attention_mask, segment_size=None)

        
        # try:
        label_hidden_states = self.hier_encoder(input_ids=label_input_ids, attention_mask=label_attention_mask, segment_size=None)
        input_fusion_states = self.baf(input_hidden_states[0], label_hidden_states[1])
        
        # except torch.cuda.OutOfMemoryError:
        #     print("+++ Skipping fusion")
        #     input_fusion_states = None
        
        input_fusion_states = None
        
        if input_fusion_states is not None:
            logits = self.dropout(self.classifier_fc(input_hidden_states[1] + input_fusion_states))
        else:
            logits = self.dropout(self.classifier_fc(input_hidden_states[1]))
    
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
    
        return TextClassifierOutput(loss=loss, logits=torch.sigmoid(logits), hidden_states=input_fusion_states)

bert = AutoModel.from_pretrained(model_src) #, cache_dir='~/HDD/LSI-Cache')
bert.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
hier_bert = HierBert(bert)
baf = BilinearAttentionFusion(768, 512)
model = BAFForTextClassfication(hier_bert, baf, len(label_vocab), label_weights=label_weights).cuda()

# Compute macro F1 scores
def compute_metrics(p, threshold=0.5):
    metrics = {}
    preds = (p.predictions > threshold).astype(float)
    refs = p.label_ids
    print(preds.shape, refs.shape)
    metrics['prec'] = precision_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['rec'] = recall_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['f1'] = f1_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    return metrics

# Different layers have different learning rates, here we set the learning rates for each layer
def AdamWLLRD(model, bert_lr=3e-5, intermediate_lr=5e-4, top_lr=1e-3, wd=1e-2):
    opt_params = []
    named_params = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert = ["hier_encoder.bert_encoder.embeddings", "hier_encoder.bert_encoder.encoder"]    
    intermediate = ["hier_encoder.bert_encoder.pooler", "hier_encoder.segment_encoder"]
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
sch = transformers.get_constant_schedule_with_warmup(opt, num_warmup_steps=1000)

training_args = TrainingArguments(
    output_dir=os.path.join(root, output_fol),
    overwrite_output_dir=False,
    do_train=False,
    do_eval=False,
    do_predict=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=16,
    eval_accumulation_steps=None,
    learning_rate=None,
    weight_decay=None,
    num_train_epochs=15,
    logging_strategy='epoch',
    logging_first_step=False,
    save_strategy='epoch',
    save_total_limit=1,
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
    label_names=['labels'],
    use_cpu=False
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
    test_results = trainer.evaluate(eval_dataset=dataset['test'], ignore_keys=['hidden_states'])
    print(test_results)
    trainer.save_metrics('test', test_results)
    
if training_args.do_predict:
    model.load_state_dict(torch.load(os.path.join(root, output_fol, "pytorch_model.bin"), map_location='cuda'))
    predictions, label_ids, results = trainer.predict(test_dataset=dataset['test'], ignore_keys=['hidden_states'])
    np.save(os.path.join(root, output_fol, "predictions.npy"), predictions)
    np.save(os.path.join(root, output_fol, "label_ids.npy"), label_ids)
    print(results)
    trainer.save_metrics('test', results)
