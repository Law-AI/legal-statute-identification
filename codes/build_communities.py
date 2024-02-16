import json
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

folder = sys.argv[1]

with open(os.path.join(folder, "label_descriptions.json")) as fr:
    L = json.load(fr)['data']
    


Lt = [l['title'] + ":\n" + '\n'.join(l['text']) for l in L]
print(len(Lt))
print(Lt[0])

# vectorizer = TfidfVectorizer(max_df=len(Lt)//2, min_df=len(Lt)//10)

vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_df=0.4, min_df=0.05)

Le = vectorizer.fit_transform(Lt)
lv = vectorizer.get_feature_names_out()
print(len(lv), Le.shape, type(Le))
sim_matrix = cosine_similarity(Le)


# vectorizer = SentenceTransformer("sentence-transformers/all-distilroberta-v1", device='cuda')
# vectorizer.max_seq_length = 512


# Le = vectorizer.encode(Lt, convert_to_tensor=True, show_progress_bar=True, device='cuda')

# sim_matrix = util.pytorch_cos_sim(Le, Le)
print(sim_matrix.shape)

edges = []
for i in range(sim_matrix.shape[0]):
    for j in range(sim_matrix.shape[1]):
        if j == i: continue
        if sim_matrix[i,j] > 0.49:
            edges.append([i, j])
print(len(edges))

with open(os.path.join(folder, "ladan_edges.json"), 'w') as fw:
    json.dump(edges, fw, indent=4)

# community detection
comms = {i: [i] for i in range(len(Lt))}
print(len(comms))
for e in tqdm(edges):
    #print(e)
    v1, v2 = e[0], e[1]
    c1, c2 = comms[v1], comms[v2]
    #print(c1, c2)
    if c1 == c2:
        continue
    assert len(set(c1) & set(c2)) == 0
    c3 = c1 + c2
    #print(c3)
    for v, c in comms.items():
        if c == c1 or c == c2:
            comms[v] = c3
    #print(len(comms))
            #print(v)
    #input()
            
commset = []
for v, c in tqdm(comms.items()):
    if c not in commset:
        commset.append(c)
        
print(commset)

label2comm = {}
for i, c in enumerate(commset):
    for v in c:
        label2comm[v] = i

print(len(label2comm), len(set(label2comm.values())))
print(label2comm)
with open(os.path.join(folder, "label2community.json"), 'w') as fw:
    json.dump(label2comm, fw, indent=4)
    

        

        






