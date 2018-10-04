import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model(final)'%(loss_model))


dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word]]

==========================================================================
"""
f=open("word_analogy_dev.txt",'r')
g=open("gen_file_dev_nce.txt",'w+')
if f.mode == 'r':
    fl = f.readlines()
    for x in fl:
        x = x.replace('"','')
        x=x.replace('\n','')
        e,c = x.split('||')
        ep = e.split(',')
        cp=c.split(',')
        weplist = []
        wcplist = []
        for i in ep:
            wep1, wep2 = i.split(':')
            weplist.append(wep1)
            weplist.append(wep2)
        for i in cp:
            wcp1, wcp2 = i.split(':')
            wcplist.append(wcp1)
            wcplist.append(wcp2)
        veplist=[]
        vcplist=[]
        for i in weplist:
            veplist.append(embeddings[dictionary[i]])
        for i in wcplist:
            vcplist.append(embeddings[dictionary[i]])
        deplist = []
        deplist.append(veplist[0] - veplist[1])
        deplist.append(veplist[2] - veplist[3])
        deplist.append(veplist[4] - veplist[5])
        depavg = np.mean(deplist,axis=0)
        dcplist =[]
        #First calculated using the difference vectors given in the question paper of assigments. However results were not satisfying.

        # dcplist.append(cosine_similarity(np.atleast_2d(vcplist[1] - vcplist[0]),np.atleast_2d(depavg)))
        # dcplist.append(cosine_similarity(np.atleast_2d(vcplist[3] - vcplist[2]),np.atleast_2d(depavg)))
        # dcplist.append(cosine_similarity(np.atleast_2d(vcplist[5] - vcplist[4]),np.atleast_2d(depavg)))
        # dcplist.append(cosine_similarity(np.atleast_2d(vcplist[7] - vcplist[6]),np.atleast_2d(depavg)))

        #Then a used a variation of the above difference vector formula, where I added A-B+C and it should be equal to the D vector.
        #A-B+C=D. I compared cosine similarity of A-B+C and D. Results were satisfying.

        dcplist.append(cosine_similarity(np.atleast_2d(depavg+vcplist[0]), np.atleast_2d(vcplist[1])))
        dcplist.append(cosine_similarity(np.atleast_2d(depavg+vcplist[2]), np.atleast_2d(vcplist[3])))
        dcplist.append(cosine_similarity(np.atleast_2d(depavg+vcplist[4]), np.atleast_2d(vcplist[5])))
        dcplist.append(cosine_similarity(np.atleast_2d(depavg+vcplist[6]), np.atleast_2d(vcplist[7])))
        dcplist=np.reshape(dcplist,[1,-1])
        l=np.argsort(dcplist)
        word_pair_most=cp[l[0][-1]]
        word_pair_least=cp[l[0][0]]

        g.write('"'+cp[0]+'"'+' "'+cp[1]+'"'+' "'+cp[2]+'"'+' "'+cp[3]+'" "'+ word_pair_least+'" "'+word_pair_most+'"\n')

g.close()
f.close()
# Top 20 words similar to {first,american,would}
vf = embeddings[dictionary["first"]]
va= embeddings[dictionary["american"]]
vw= embeddings[dictionary["would"]]
ansF=[]
ansA=[]
ansW=[]
print("Top 20 Words:")
for word,word_id in dictionary.items():
    v1 = embeddings[word_id]
    ansF.append((cosine_similarity(np.atleast_2d(v1), np.atleast_2d(vf)),word))
    ansA.append((cosine_similarity(np.atleast_2d(v1), np.atleast_2d(va)),word))
    ansW.append((cosine_similarity(np.atleast_2d(v1), np.atleast_2d(vw)),word))
ansF.sort(reverse=True)
ansA.sort(reverse=True)
ansW.sort(reverse=True)
print("first:")
a,b=zip(*ansF[1:21])
print(b)
print("american:")
a,b=zip(*ansA[1:21])
print(b)
print("would:")
a,b=zip(*ansW[1:21])
print(b)
