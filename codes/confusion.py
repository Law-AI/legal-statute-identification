import numpy as np 
import json
import os
from tqdm import tqdm
from collections import defaultdict
import sys
from sklearn.metrics import f1_score
import statistics
import random


random.seed(42)

model = sys.argv[1]
tfidf = '_tfidf' if sys.argv[2] == 'True' else ''
output_fol = f'/home/shounak/HDD/SIGIR2024/ILSI/{model}'
root = '/home/shounak/LSI/ILSI'

print(tfidf)

gs = np.load(os.path.join(output_fol, 'label_ids.npy'))
preds = (np.load(os.path.join(output_fol, 'predictions.npy')) > 0.5).astype(float)
F = f1_score(gs, preds, average=None)
microF = f1_score(gs, preds, average='micro')

if not os.path.exists(os.path.join(root, f'confusion-{model}2.npy')):
    cf = np.zeros((gs.shape[1], gs.shape[1]))
    # nmis = preds.sum(axis=1) - (gs * preds).sum(axis=1) 
    for i in tqdm(range(gs.shape[0])):
        for j in range(gs.shape[1]):
            for k in range(gs.shape[1]):
                if j == k: continue
                if gs[i, j] == 1 and gs[i, k] == 0 and preds[i, j] == 0 and preds[i, k] == 1:
                    # cf[k, j] += 1
                    cf[j, k] += 1
                    
                # if gs[i, j] == 0 and gs[i, k] == 1 and preds[i, j] == 1 and preds[i, k] == 0:
                #     cf[k, j] += 1/nmis
                    # cf[j, k] += 1
    cf = cf / np.expand_dims(cf.sum(axis=1), 1)
    # ll = cf.tolist()
    # with open(os.path.join(root, 'confusion-ilb.npy'), 'w') as fd:
    #     json.dump(ll, fd, indent=4)
    np.save(os.path.join(root, f'confusion-{model}2.npy'), cf)
else:    
    # with open(os.path.join(root, 'confusion-ilb.json'), 'r') as fd:
    #     ll = json.load(fd)
    cf = np.load(os.path.join(root, f'confusion-{model}2.npy'))

# weights = cf.sum(axis = 0) + 1e-6
# print(weights)
weights = gs.sum(axis=0)

with open(os.path.join(root, f'label2community{tfidf}.json')) as fd:
    l2cstr = json.load(fd)

l2c = dict()
for l in l2cstr:
    l2c[int(l)] = l2cstr[l]

c2l = dict()
for l in l2c:
    if l2c[l] in c2l:
        c2l[l2c[l]].append(int(l))
    else:
        c2l[l2c[l]] = [int(l)]
        
clens = [len(v) for v in c2l.values()]

MUQ, MUR, MUR2, MQ, MR, MR2 = [], [], [], [], [], []
for X in tqdm(range(1)):
    # arr = list(range(100))
    # random.shuffle(arr)
    # c2lnew = defaultdict(int)
    # l2cnew = defaultdict(int)

    # i = 0
    # for j,l in enumerate(clens):
    #     c2lnew[j] = arr[i:i+l]
    #     i += l
        
    # for k,v in c2lnew.items():
    #     for vv in v:
    #         l2cnew[vv] = k
            
    # l2c = l2cnew
    # c2l = c2lnew

    output = dict()
    for i in range(gs.shape[1]):
        comm = l2c[i]
        s1 = 0
        n1 = len(c2l[comm]) - 1
        n2 = gs.shape[1] - n1 - 1
        s2 = []
        c2lscore = defaultdict(list)
        for j in range(gs.shape[1]):
            if i == j: continue
            if j in c2l[comm]: s1 += (cf[i][j]) 
            else:
                s2.append(cf[i][j]) 
        
        # for key in c2lscore:
        #     c2lscore[key] = sum(c2lscore[key]) / len(c2lscore[key])
        
        # s2 = statistics.mean(list(c2lscore.values()))
        
            
        
        s1 = s1 / n1 if n1 > 0 else 0
        s2m = sum(sorted(s2, reverse=True)[:n2]) / n2 if n2 > 0 else 0
        # s1 = max(s1) if n1 > 0 else 0
        # s2 = max(s2) if n2 > 0 else 0
        s2r = [sum(random.sample(s2, n1)) / n1 if n1 > 0 else 0 for _ in range(20)]
        
        output[i] = (s1,s2m,s2r)

    # with open(os.path.join(root, f'community_outputs_{model}{tfidf}.json'), 'w') as fd:
    #     json.dump(output, fd, indent=4)

    # with open(os.path.join(root, f'community_outputs_{model}{tfidf}.json'), 'r') as fd:
    #     scores = json.load(fd)

    # scores = output
    c1 = defaultdict(list)
    c2m = defaultdict(list)
    c2r = [defaultdict(list) for _ in range(20)]
    ss1 = []
    ss2m = []
    ss2r = [[] for _ in range(20)]
    f1 = defaultdict(list)
    for key in output:
        comm = l2c[key]
        if len(c2l[comm]) != 1:
            s1, s2m, s2r = output[key]
            c1[comm].append(s1)
            c2m[comm].append(s2m)
            ss1.append(s1)
            ss2m.append(s2m)
            f1[comm].append(F[int(key)])
            for a in range(20):
                c2r[a][comm].append(s2r[a])
                ss2r[a].append(s2r[a])

    muQ = statistics.mean(ss1)
    muRm = statistics.mean(ss2m)
    ss2r = [statistics.mean(x) for x in ss2r]
    muRr = statistics.mean(ss2r)
    muRr_sd = statistics.stdev(ss2r)

    mQ, mRm, mRrx, mF = [], [], [[] for _ in range(20)], []
    comms= []
    for comm in c1:
        mQ.append(statistics.mean(c1[comm]))
        mRm.append(statistics.mean(c2m[comm]))
        mF.append(statistics.mean(f1[comm]))
        for a in range(20):
            mRrx[a].append(statistics.mean(c2r[a][comm]))
        comms.append(comm)
    
    print({'comm': comms, 'Q': mQ, 'R': mRm, 'F': mF})
    
    with open(os.path.join(output_fol, f"confusion{tfidf}.json"), 'w') as fw:
        json.dump({'comm': comms, 'Q': mQ, 'R': mRm, 'F': mF}, fw, indent=4)

    exit(0)
            
    mQ = statistics.mean(mQ)
    mRm = statistics.mean(mRm)
    mRrx = [statistics.mean(x) for x in mRrx]
    mRr = statistics.mean(mRrx)
    mRr_sd = statistics.stdev(mRrx)

    # with open(os.path.join(output_fol, f"confusion{tfidf}.txt"), 'w') as fw:
    print(model, "=====>")
    print(f"Micro-Q: {muQ:.04f}, Micro-R (all): {muRm:.04f}, Micro-R (random): {muRr:.04f} +/- {muRr_sd:.04f}")
    print(f"Macro-Q: {mQ:.04f}, Macro-R (all): {mRm:.04f}, Macro-R (random): {mRr:.04f} +/- {mRr_sd:.04f}")
    #     print(f"Micro-Q: {muQ:.04f}, Micro-R (all): {muRm:.04f}, Micro-R (random): {muRr:.04f} +/- {muRr_sd:.04f}", file=fw)
    #     print(f"Macro-Q: {mQ:.04f}, Macro-R (all): {mRm:.04f}, Macro-R (random): {mRr:.04f} +/- {mRr_sd:.04f}", file=fw)
    
    MUQ.append(muQ)
    MUR.append(muRm)
    MUR2.append(muRr)
    MQ.append(mQ)
    MR.append(mRm)
    MR2.append(mRr)
    
# print(model, "=====>")
# print(f"Micro-Q: {statistics.mean(MUQ):.04f} +/- {statistics.stdev(MUQ):.04f}, Micro-R (all): {statistics.mean(MUR):.04f}, +/- {statistics.stdev(MUR):.04f}, Micro-R (random): {statistics.mean(MUR2):.04f} +/- {statistics.stdev(MUR2):.04f}")
# print(f"Macro-Q: {statistics.mean(MQ):.04f} +/- {statistics.stdev(MQ):.04f}, Macro-R (all): {statistics.mean(MR):.04f}, +/- {statistics.stdev(MR):.04f}, Micro-R (random): {statistics.mean(MR2):.04f} +/- {statistics.stdev(MR2):.04f}")
  
        
        
         

# ss1, ss2 = [], []
# for key in scores:
#     comm = l2c[int(key)]
#     if len(c2l[comm]) != 1:
#         c1[comm].append(scores[key][0])
#         ss1.append(scores[key][0])
#         f1[comm].append(F[int(key)])
#         c2[comm].append(scores[key][1])
#         ss2.append(scores[key][1])

# l = []
# s1 = []
# s2 = []
# with open(os.path.join(output_fol, f"confusion{tfidf}.txt"), 'w') as fw:
#     print("Micro", "-", sum(ss1)/len(ss1), sum(ss2)/len(ss2), microF,  sep='\t', file=fw)
#     for key in c1:
#         print(key, len(c1[key]), sum(c1[key]) / len(c1[key]), sum(c2[key]) / len(c2[key]), sum(f1[key]) / len(f1[key]), sep='\t', file=fw)
#         if len(c1[key]) > 1:
#             l.append(len(c1[key]))
#             s1.append(sum(c1[key]) / len(c1[key]))
#             s2.append(sum(c2[key]) / len(c2[key]))
#     print('', sum(l)/len(l), sum(s1)/len(s1), sum(s2)/len(s2), F.mean(), sep='\t', file=fw)


    



    

            
