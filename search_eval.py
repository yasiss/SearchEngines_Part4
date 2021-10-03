#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install --upgrade pip


# In[2]:


pip install metapy pytoml


# In[3]:


import metapy
idx = metapy.index.make_inverted_index('config.toml')
metapy.log_to_stderr()


# In[4]:


import math
import sys
import time
import metapy
import pytoml


# In[ ]:


'''
ana = metapy.analyzers.load('config.toml')
doc = metapy.index.Document()
doc.content("I said that I can't believe that it only costs $19.95!")
print(ana.analyze(doc))
'''


# In[5]:


# Examine number of documents
idx.num_docs()
# Number of unique terms in the dataset
idx.unique_terms()
# The average document length
idx.avg_doc_length()
# The total number of terms
idx.total_corpus_terms()


# In[6]:


def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    """
    return metapy.index.OkapiBM25(k1=1.25,b=0.75,k3=500)


# In[8]:


if __name__ == '__main__':
    #if len(sys.argv) != 2:
     #   print("Usage: {} config.toml".format(sys.argv[0]))
     #   sys.exit(1)

    #cfg = sys.argv[1]
    cfg= 'config.toml'
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    query = metapy.index.Document()
    ndcg = 0.0
    num_queries = 0

    print('Running queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            ndcg += ev.ndcg(results, query_start + query_num, top_k)
            num_queries+=1
    ndcg= ndcg / num_queries
            
    print("NDCG@{}: {}".format(top_k, ndcg))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))

