import sys, time, logging, os, shutil
import re
import pickle as pkl
import pandas as pd
import numpy as np
import multiprocess as mp
from tqdm.notebook import tqdm_notebook
import requests, json
import bs4
from bs4 import BeautifulSoup
import spacy
import codecs
from tqdm import tqdm

spacy.prefer_gpu()
spacy.require_gpu()

with open("indian_names.txt") as fr:
    indian_names = set(fr.read().strip().split('\n'))
    
nlp = spacy.load("en_legal_ner_trf")
nlp.max_length = int(1e9)

files = ["ILSI/train.json", "ILSI/dev.json", "ILSI/test.json"]


indian_names = [r'\b{}\b'.format(re.escape(i)) for i in indian_names] # match names only at word boundaries
match_string = '|'.join(indian_names)
vowels = set(['a', 'e', 'i', 'o', 'u'])

for f in files:
    print("Processing", f, "---")
    with open(f) as fr:
        D = json.load(fr)
    for i,doc in enumerate(tqdm(D['data'])):
        #if i == 10:
            #break
        text = []
        for sent in doc['text']:
            xsent = sent.lower()
            v = re.findall("a|e|i|o|u", xsent)
            if len(v) / len(xsent) < 0.2:
                continue
            psent = nlp(sent)
            fsent = psent.text
            for ent in psent.ents:
                if ent.label_ in ["PETITIONER", "RESPONDENT", "JUDGE", "LAWYER", "WITNESS", "OTHER_PERSON"]:
                    fsent = fsent.replace(ent.text, '[ENTITY]')
                elif ent.label_ == 'PROVISION':
                    fsent = fsent.replace(ent.text, '[SECTION]')
                elif ent.label_ == 'STATUTE':
                    fsent = fsent.replace(ent.text, '[ACT]')
                elif ent.label_ == 'PRECEDENT':
                    fsent = fsent.replace(ent.text, '[PRECEDENT]')
            fsent, nsub = re.subn(match_string, '[ENTITY]', fsent)
            text.append(fsent)
        doc['text'] = text
    with open(f.split('.')[0] + "2.json", 'w') as fw:
        json.dump(D, fw, indent=4)
            
            
