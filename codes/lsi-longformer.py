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
from sklearn.metrics import precision_score, recall_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

root = sys.argv[1]              # Dataset directory
model_src = sys.argv[2]         # Model name/directory  
output_fol = sys.argv[3]        # Output Folder

max_seq_len = 4096              # Max sequence length

CACHE_DIR = "~/HDD/LSI-Cache"   # Cache location

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


dataset = load_dataset('json', data_files={'train': os.path.join(root, "train2.json"), 
                                           'dev': os.path.join(root, "dev2.json"), 
                                           'test': os.path.join(root, "test2.json")}, 
                       field='data', 
                       cache_dir=CACHE_DIR)
print(dataset)
dataset = dataset.map(schema.encode_example, features=schema)

config = AutoConfig.from_pretrained(model_src), #cache_dir=CACHE_DIR)

tokenizer = AutoTokenizer.from_pretrained(model_src)
special_tokens = {'additional_special_tokens': ['<ENTITY>', '<ACT>', '<SECTION>']}
tokenizer.add_special_tokens(special_tokens)
tokenizer.add_tokens(special_tokens['additional_special_tokens'])

assert tokenizer.pad_token_id is not None
assert tokenizer.bos_token_id is not None
assert tokenizer.eos_token_id is not None

print('Filtering all empty rows')
dataset = dataset.filter(lambda example: len(example['text']) != 0)
print(dataset)


dataset = dataset.map(lambda example: tokenizer(list(example['text']), 
                        return_token_type_ids=False), 
                      batched=False)

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print(dataset['train'].features)

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

label_weights = generate_label_weights()

class TextClassifierOutput(ModelOutput):
    loss = None
    logits = None
    hidden_states = None

# Longformer encoder + classification head on top of [CLS] token
class LongformerForTextClassification(nn.Module):
    def __init__(self, encoder, num_labels, label_weights=None, drop=0.5):
        super().__init__()
        
        self.hidden_size = encoder.config.hidden_size
        self.num_labels = num_labels
        self.encoder = encoder
        self.classifier_fc = nn.Linear(encoder.config.hidden_size, num_labels)
        
        if label_weights is None:
            label_weights = torch.ones(num_labels)  # For the rare labels
        self.loss_fct = nn.BCEWithLogitsLoss(label_weights)
        
        self.dropout = nn.Dropout(drop)
    
    def gradient_checkpointing_enable(self):
        self.hier_encoder.gradient_checkpointing_enable()

    def forward(self, input_ids=None, 
                attention_mask=None, 
                global_attention_mask=None, 
                labels=None):
        hidden = self.dropout(self.encoder(input_ids=input_ids, 
                                           attention_mask=attention_mask,
                                           global_attention_mask=global_attention_mask).last_hidden_state[:, 0, :])
        logits = self.dropout(self.classifier_fc(hidden))
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        
        return TextClassifierOutput(loss=loss, logits=torch.sigmoid(logits), hidden_states=hidden) 



longformer = AutoModel.from_pretrained(model_src) #, cache_dir=CACHE_DIR)
longformer.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = LongformerForTextClassification(longformer, 
                                        len(label_vocab), 
                                        label_weights=label_weights).cuda()

# Creates the tensors to feed to the model, i.e. the data collator
def collate_fn(examples):
    input_ids = torch.zeros(len(examples), max_seq_len, dtype=torch.long).fill_(tokenizer.pad_token_id)
    labels = torch.zeros(len(examples), len(label_vocab))
    
    for exp_idx, exp in enumerate(examples):
        if len(exp['input_ids']) > 1:
            exp_text = torch.cat([s[1:-1] for s in exp['input_ids']], dim=0)[:max_seq_len-2]
        else: 
            exp_text = exp['input_ids'][0][:max_seq_len-2]
        
        input_ids[exp_idx, 1:len(exp_text)+1] = exp_text
        input_ids[exp_idx, 0] = tokenizer.bos_token_id
        input_ids[exp_idx, len(exp_text)] = tokenizer.eos_token_id
        
        labels[exp_idx].scatter_(0, exp['labels'], 1.)
        
    attention_mask = input_ids != tokenizer.pad_token_id
    global_attention_mask = input_ids == tokenizer.bos_token_id
    
    return BatchEncoding({'input_ids': input_ids, 
                          'attention_mask': attention_mask, 
                          'labels': labels, 
                          'global_attention_mask': global_attention_mask})


# Computes macro F1 score as reported in the paper
def compute_metrics(p, threshold=0.5):
    metrics = {}
    preds = (p.predictions > threshold).astype(float)
    refs = p.label_ids
    metrics['prec'] = precision_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['rec'] = recall_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    metrics['f1'] = f1_score(refs, preds, average='macro', labels=list(label_vocab.values()))
    return metrics


# Different layers have different learning rates, here we set the learning rates for each layer
def AdamWLLRD(model, bert_lr=1e-4, intermediate_lr=1e-3, top_lr=1e-3, wd=1e-2):
    opt_params = []
    named_params = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert = ["encoder.embeddings", "encoder.encoder", "encoder.pooler"]    
    intermediate = []
    top = ["classifier_fc"]

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
sch = transformers.get_linear_schedule_with_warmup(opt, 
                                                   num_warmup_steps=800, 
                                                   num_training_steps=23000)



training_args = TrainingArguments(
    output_dir=output_fol,
    overwrite_output_dir=False,
    do_train=True,
    do_eval=False,
    do_predict=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
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
    dataloader_num_workers=8,
    load_best_model_at_end=False,
    metric_for_best_model='f1',
    greater_is_better=True,
    group_by_length=False,
    dataloader_pin_memory=True,
    resume_from_checkpoint=False,
    gradient_checkpointing=False,
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
    _, _, metrics = trainer.train(ignore_keys_for_eval=['hidden_states'], resume_from_checkpoint=True)
    trainer.save_model()
    trainer.save_metrics('train', metrics)
if training_args.do_eval:
    model.load_state_dict(torch.load(os.path.join(output_fol, "pytorch_model.bin"), map_location='cuda'))
    test_results = trainer.evaluate(eval_dataset=dataset['test'], ignore_keys=['hidden_states'])
    print(test_results)
if training_args.do_predict:
    predictions, label_ids, results = trainer.predict(test_dataset=dataset['test'], ignore_keys=['hidden_states'])
    np.save(os.path.join(output_fol, "predictions.npy"), predictions)
    np.save(os.path.join(output_fol, "label_ids.npy"), label_ids)
    trainer.save_metrics('test', results)
    print(results)
