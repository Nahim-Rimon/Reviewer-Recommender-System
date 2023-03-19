"""
Created on Fri Apr  1 22:22:52 2022
@author: NahimAhmed
"""


import pandas as pd
import numpy as np
import os
#import requests
import nltk
import string
import itertools
from collections import Counter
from nltk import everygrams
from nltk.stem import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords
#from nltk.tokenize import RegexpTokenizer
stop_words = set(stopwords.words('english'))

#from Remove_unexpected_char import remove_unexpected_char


################ Global Parameters ########################
path1 = r'.\SampleTPCMember'
path2 = r'.\Track1 TPC Member Files'
path3 = r'.\KeyPhrasesWithWeight'
path4 = r'.\ConferenceReviewerData'

EPSILON = 1e-07
###########################################################

def FindProbability (confidence, similarity, values):
    
    count = 0
    for item in values:
        #g = float("{:.1f}".format(item[4]))
        g = round (item[4], 2)
        similarity = round (similarity, 2)
        print (item[5], g)
        if item[3] == confidence and g >= similarity:
            #print (item)
            count += 1
    
    #print (count / len(values))
    return count / len(values)

def FindCount (confidence, similarity, values):
    count = 0
    for item in values:
        g = round (item[4], 2)
        similarity = round(similarity, 2)
        
        if item[3] == confidence and g >= similarity:
            #print (item)
            count += 1
    
    return count

def threshold_print(pl):
    filename = "similarity_thres_5.csv"
    filepath = os.path.join(path4, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    f = open(filepath, 'w', encoding='utf=8')
    f.write('Threshold Point, ' + 'Upper_C_Prob,' + 'Lower_C_Prob,' + 'Ratio' + '\n')
    
    for item in pl:
        f.write(str(item[0]) + ', ' + str(item[1]) + ', ' + str(item[2]) + ', ' + str(item[3]) + '\n')
        


def FindSimilarityThreshold(target_confidence):
    
    Rdata = pd.read_csv(f'{path4}/SimilarityAndOtherInfo_200.csv')
    
    max_r = 0.0 #maximum ratio
    sim_thr = 0.0 #similarity threshold
    i = 0.00
    pl = []
    
    while i <= 1.01:
        threshold_print(pl)
        p_tgt = 0.0
        for j in range (target_confidence, 6):
            c_tgt = FindCount(j, i, Rdata.values) 
            p_tgt += c_tgt / len(Rdata.values)
        #print (i, c_tgt, p_tgt)
        
        p_j = 0.0
        for j in range (1, target_confidence):
            c_j = FindCount(j, i, Rdata.values) 
            p_j += c_j / len(Rdata.values)
        
        if p_tgt > 0.2 and p_j > 0:
            p = (i, p_tgt, p_j, p_tgt / p_j)
            pl.append(p)
            r = p_tgt / p_j
            if r > max_r:
                max_r = r
                sim_thr = i
            
            i += 0.01
            i = round(i, 2)
        
        else:
            return sim_thr
        
    #print(pl)

###########################################################
#               Main Function                             #
###########################################################

if __name__ == '__main__':
    
    similarity_threshold = FindSimilarityThreshold(5)
    print(similarity_threshold)
    
    
