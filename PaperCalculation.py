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
# Datasets may be vary after the project completed.
path1 = r'.\SampleTPCMember'
path2 = r'.\Track1 TPC Member Files'
path3 = r'.\KeyPhrasesWithWeight'
path4 = r'.\ConferenceReviewerData'

EPSILON = 1e-07

###########################################################
#               Class Phrase                              #
###########################################################

class Phrase:
    def __init__ (self, phrase):
        self.Phrase = phrase
        self.Trm_Frq = []
        self.Phrz_Cnt = []
        self.tfidf = 0.0

    def __str__(self):
        return "{} | TF {} | TC {} | TFIDF {}".format(self.Phrase, self.Trm_Frq,
                                                      self.Phrz_Cnt, self.tfidf)
    
    def __repr__(self):
        return str(self)
    
    # magic method to compare two phrases
    def __eq__(self, other):
        if self.Phrase == other: 
            return True
        
    def TotalFreq (self):
        cnt = 0
        for item in self.Trm_Frq:
            cnt += item
        return cnt


###########################################################
#               Class FindPhrases                         #
###########################################################

class FindPhrases:  
    
    def __init__(self, text, select = 1):
        self.text = text
        
    def __str__(self):
        return "{}\n".format(self.text)

    def __repr__(self):
        return str(self)
    
    def ExtractPhrases (self):
        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        
        # stem all the words
        words = nltk.word_tokenize(self.text)
        for i, ele in enumerate (words):
            words[i] = ps.stem(ele)
        
        tagged_words = nltk.pos_tag_sents(nltk.word_tokenize(word) for word in words)
        #print (tagged_words)
        
        # make noun phrases
        grammar = r'KT: { (<NN.*>+ <JJ.*>?)|(<JJ.*>? <NN.*>+)}'
        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(word)) for word in tagged_words))
        #print (all_chunks)
        
        # join constituent chunk words into a single chunked phrase
        candidates = [' '.join(word for word, pos, chunk in group).lower() for key, group in itertools.groupby(all_chunks, lambda word__pos__chunk: word__pos__chunk[2] != 'O') if key]
        #print (candidates)
        Phrases = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
        y = Phrases[len(Phrases)-1].split()
        for item in y:
            if item == '…':
                y.remove('…')
        Phrases[len(Phrases)-1] = ' '.join(word for word in y)
        #print (Phrases)
        
        return Phrases
        
    def FindPhrasesAfterNGram (self, Phrases):
        # use n-gram technique to extract all possible keyphrases
        fphrases = []
        for item in Phrases:
            lw = list(everygrams(nltk.word_tokenize(item), 1, 4))
            for ele in lw:
                fphrases.append(' '.join(ele))
        #print(fphrases)
        return fphrases
    

###########################################################
#               Class CalculateWeight                     #
###########################################################

class CalculateWeight:
    def __init__(self, fphrases, n):
        self.FPhrases = fphrases
        self.N = n
        
    def __str__(self):
        return "{}\n".format(self.FPhrases)

    def __repr__(self):
        return str(self)
    
    def FindPhraseInList (self, phrase, pList):
        for item in pList:
            if item.Phrase == phrase:
                return item
        return None
    
    def FindPhraseStat (self):
        AllPhraseList = []
        for wlst in self.FPhrases:
            #print (wlst)
            counts = Counter(wlst)
            #print (counts)
            for k in counts:
                pc = self.FindPhraseInList(k, AllPhraseList)                                    
                if pc != None:
                    pc.Trm_Frq.append(counts[k])
                    pc.Phrz_Cnt.append(len(wlst))
                else:
                    pc = Phrase(k)
                    pc.Trm_Frq.append(counts[k])
                    pc.Phrz_Cnt.append(len(wlst))
                    AllPhraseList.append(pc)
        return AllPhraseList

    def CalculateTotalTF (self, APL):
        cnt = 0
        for item in APL:
            cnt += item.TotalFreq()
        return cnt

   # def FindIDF (self, Df, select = 1):
     #   if select == 1:
     #       return 1
      #  elif select == 2:
      #      return np.log(self.N/(Df+1))
        
    def CalcualteTFIDF (self, SPL):
        for item in SPL:
            tf = 0
            idf = 1
            tfidf = 0
            for i in range (len(item.Trm_Frq)):
                tf += item.Trm_Frq[i]/item.Phrz_Cnt[i]
                tfidf = tf * idf 
            item.tfidf = tfidf 
           

###########################################################
#               Class CalculateSimilarity                 #
###########################################################
# All similarity related algorithms can be implemented under this class
class CalculateSimilarity:
    def Cosine(self, x, y):
        dot_products = np.dot(x, y.T)
        norm_products = np.linalg.norm(x) * np.linalg.norm(y)
        return dot_products / (norm_products + EPSILON)
    
    def Jaccard(self, p, q):
        intersection = len(list(set(p).intersection(q)))
        union = (len(p) + len(q)) - intersection
        return float(intersection) / union


###########################################################
#               Class Utility                             #
###########################################################

class Utility:
    def SortedPhraseList (self, SPL):
        SrtLst = sorted(SPL, key=lambda x: x.tfidf, reverse=True)
        return SrtLst
    
    def TrimList (self, APL, threshold):
        APL = [item for item in APL if item.tfidf > threshold]
        return APL
    
    def SlectTopNPhraseWeight (self, SPL, topn):
        TopNPhzWght = []
        SrtPhzLst = self.SortedPhraseList(SPL)
        for i in range (len(SrtPhzLst)):
            TopNPhzWght.append (SrtPhzLst[i]) 
            if i >= topn - 1:
                break;
        return TopNPhzWght
        
    def PhraseWeightList (self, SPL):
        PhrzWghtLst = []
        for item in SPL:
            PhrzWghtLst.append((item.Phrase, item.tfidf))
        return PhrzWghtLst
    
    def PrintPhraseAndWeight_paper (self, PL, Name):
        filename = f"{Name}.csv"
        filepath = os.path.join(path3, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        f = open(filepath, 'w', encoding='utf=8')
        f.write('Keyphrase, ' + 'Weight' + '\n')
        
        for item in PL:
            f.write (item[0] + ', ' + str(item[1]) + '\n')
            

###########################################################
#               General Functions                         #
###########################################################
  
def AssignedPapersVsKeyPhrases ():
    
    threshold = 0.2
    topn = 200
    AllPaperPhrases = {}
    Rdata = pd.read_csv(f'{path4}/ReviewersData.csv')
    
    for ele in Rdata.values:
        
        paper_id = ele[2]

        TitleData = ele[4]
        #print (TitleData)
        
        AbstractData = ele[5]
        #print (AbstractData)
        TitleAndAbstractData = TitleData + '. ' + AbstractData
        N = len(TitleAndAbstractData)

        extract_candidate = FindPhrases(TitleAndAbstractData)
        
        
        Phrases = extract_candidate.ExtractPhrases()

        PhrasesList1 = extract_candidate.FindPhrasesAfterNGram(Phrases)
        
    
        calw = CalculateWeight (PhrasesList1, N)
                    
                    # To find term frequency (TF) and document frequency (df) of various phrases
        AllPhrzLst = calw.FindPhraseStat()
    
        calw.CalcualteTFIDF(AllPhrzLst)
                    #print (SltPhrzLst)
    
        utz = Utility()
                    
        ChoiceList = ['Threshold', 'TopN']
        choice = ChoiceList[1]
                    
        if (choice == 'Threshold'):
            SltPhrzLst = utz.TrimList(AllPhrzLst, threshold)
        else:
            SltPhrzLst = utz.SlectTopNPhraseWeight(AllPhrzLst, topn)
                #print (SltPhrzLst)
                    
            # Find only phrase and weight
        PhrzWghtLst = utz.PhraseWeightList(SltPhrzLst)
        AllPaperPhrases[paper_id] = PhrzWghtLst  
    
        utz.PrintPhraseAndWeight_paper(PhrzWghtLst, paper_id)
    
###########################################################
#               Main Function                             #
###########################################################

if __name__ == '__main__':
    
    AssignedPapersVsKeyPhrases ()
    
        
    