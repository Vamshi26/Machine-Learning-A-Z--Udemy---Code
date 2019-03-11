# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:59:58 2019

@author: lalit.h.suthar
"""

'''
normalization is a process of scaling individual data samples so as to have unit norm. 
And it's a common operation for text classification or clustering. 
'''

from sklearn.preprocessing import Normalizer
'''
Normalize samples individually to unit norm.

Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of other samples so that its norm (l1 or l2) equals one.

This transformer is able to work both with dense numpy arrays and scipy.sparse matrix (use CSR format if you want to avoid the burden of a copy / conversion).

Scaling inputs to unit norms is a common operation for text classification or clustering for instance. 

For instance the dot product of two l2-normalized TF-IDF vectors is the cosine similarity of the vectors and is the base similarity metric for the Vector Space Model commonly used by the Information Retrieval community.
'''

X = [[4, 1, 2, 2],
     [1, 3, 9, 3],
     [5, 7, 5, 1]]
transformer = Normalizer().fit(X) # fit does nothing.
print(transformer.transform(X))