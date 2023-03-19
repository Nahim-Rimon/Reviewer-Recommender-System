"""
Created on Wed Jun  8 10:44:13 2022
@author: NahimAhmed
"""

import pandas as pd
import numpy as np
import os
import random
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
path5 = r'.\ArticleKeyPhrases'
path6 = r'.\Results'

EPSILON = 1e-07
SIMULATION_THRESHOLD = 0.27
REVIEWER_COUNT = 3
MAX_ASSIGNED_REVIEW = 4
# RB <- RANDOMWALK BRUTEFORCE RR <- RANDOMWALK RANDOMBRUTEFORCE
SELECTION_MODE = ['RANDOMWALK', 'BRUTEFORCE', 'RANDOMBRUTEFORCE', 'MIXED_RB', 
                  'MIXED_RR']
SELECTED_MODE = SELECTION_MODE[1]
###########################################################

###########################################################
#                      Class Article
###########################################################

class Article:
    def __init__ (self, name, phrases):
        self.Name = name
        self.Phrases = phrases
        self.ArticleAssignedTo = []
        
    def __str__(self):
        return "{} | ArticleAssignedTo {} \n".format(self.Name, 
                                                      self.ArticleAssignedTo)
    
    def __repr__(self):
        return str(self)
    
    # magic method to compare two phrases
    def __eq__(self, other):
        if self.Name == other: 
            return True
        
    def NoOfReviewerAssigned (self):
        return len(self.ArticleAssignedTo)

##############################################################
#               Class Reviewer
##############################################################

class Reviewer:
    def __init__ (self, name, phrases):
        self.Name = name
        self.Phrases = phrases
        self.ArticleAssigned = []
        
    def __str__(self):
        return "{} | Article Assigned {} \n".format(self.Name, 
                                                      self.ArticleAssigned)
    
    def __repr__(self):
        return str(self)
    
    # magic method to compare two phrases
    def __eq__(self, other):
        if self.Name == other: 
            return True
        
    def NoOfArticleAssigned (self):
        return len(self.ArticleAssigned)
    
###########################################################
#               Class CalculateSimilarity                 #
###########################################################

# All similarity related algorithms can be implemented under this class

class CalculateSimilarity:
    def Cosine(self, x, y):
        dot_products = np.dot(x, y.T)
        norm_products = np.linalg.norm(x) * np.linalg.norm(y)
        return dot_products / (norm_products + EPSILON)
    
    def Jaccard (self, x, y):
        union = x.union(y)
        intersec = x.intersection(y)
        sim = len(intersec)/len(union)
        return sim

##################################################################
#              Utility Function for Similarity Calculation
##################################################################

def FindXY (Article, Reviewer):
    X = []
    Y = []
    
    rvwPhr = Reviewer.Phrases.values.tolist()
    artPhr = Article.Phrases.values.tolist()
    
    for item in artPhr:
        X.append(item[1])
        
        find = False
        for ele in rvwPhr:
            if ele[0] == item[0]:
                Y.append(ele[1])
                find = True

        if find == False:
            Y.append(0)
    
    return X, Y


###########################################################
#            Functions for Reviewer Selection
###########################################################

def CheckReviewer (Article, Reviewer, max_rvw):
    if Reviewer.Name in Article.ArticleAssignedTo:
        return False
    
    if len(Reviewer.ArticleAssigned) < max_rvw:
        return True
    else:
        return False


def SelectByRandomWalk (Article, ReviewersList, sim_thr, max_rvw, CalSim):
    
    count = -1
    
    while count < 3 * len(ReviewersList):
        count += 1
        rint = random.randint(0, len(ReviewersList))
        
        if rint < len(ReviewersList) and rint >= 0:
            Reviewer = ReviewersList[rint]
            
            if CheckReviewer(Article, Reviewer, max_rvw) == False:
                continue
            
            X, Y = FindXY(Article, Reviewer)
            sim = CalSim.Cosine(np.array(X), np.array(Y))
            #print (Reviewer, sim)
            
            if sim >= sim_thr:
                return Reviewer 
            
    return None

    
def SelectByBruteForce (Article, ReviewersList, sim_thr, max_rvw, CalSim):
    
    for Reviewer in ReviewersList:
        if CheckReviewer(Article, Reviewer, max_rvw) == False:
            continue
        
        X, Y = FindXY(Article, Reviewer)
        sim = CalSim.Cosine(np.array(X), np.array(Y))
        #print (Reviewer, sim)
            
        if sim >= sim_thr:
            return Reviewer 
            
    return None


def SelectByRandomBruteForce (Article, ReviewersList, sim_thr, max_rvw, CalSim):
    
    rint = random.randint(0, len(ReviewersList))
    
    for i in range (rint, len(ReviewersList), 1):
        Reviewer = ReviewersList[i]
        if CheckReviewer(Article, Reviewer, max_rvw) == False:
            continue
        
        X, Y = FindXY(Article, Reviewer)
        sim = CalSim.Cosine(np.array(X), np.array(Y))
        #print (Reviewer, sim)
            
        if sim >= sim_thr:
            return Reviewer 
        
    for i in range (rint - 1, -1, -1):
        Reviewer = ReviewersList[i]
        if CheckReviewer(Article, Reviewer, max_rvw) == False:
            continue
        
        X, Y = FindXY(Article, Reviewer)
        sim = CalSim.Cosine(np.array(X), np.array(Y))
        #print (Reviewer, sim)
            
        if sim >= sim_thr:
            return Reviewer 
            
    return None

    
def SelectReviewers (Article, ReviewersList, NoRS, sim_thr, max_rvw):
    selectedReviewer = []
    
    #print (Article)
    
    CalSim = CalculateSimilarity()
    
    count = 0
    while Article.NoOfReviewerAssigned() < NoRS and count < NoRS:
        
        if SELECTED_MODE == 'RANDOMWALK':
            Reviewer = SelectByRandomWalk (Article, ReviewersList, sim_thr,
                                           max_rvw, CalSim)

            if Reviewer != None:
                 Reviewer.ArticleAssigned.append(Article.Name)
                 Article.ArticleAssignedTo.append(Reviewer.Name)
                 selectedReviewer.append(Reviewer)
            
        elif SELECTED_MODE == 'BRUTEFORCE':    
            Reviewer = SelectByBruteForce (Article, ReviewersList, sim_thr,
                                           max_rvw, CalSim)
            
            if Reviewer != None:
                Reviewer.ArticleAssigned.append(Article.Name)
                Article.ArticleAssignedTo.append(Reviewer.Name)
                selectedReviewer.append(Reviewer)
        
        elif SELECTED_MODE == 'RANDOMBRUTEFORCE':
            Reviewer = SelectByRandomBruteForce (Article, ReviewersList, sim_thr,
                                           max_rvw, CalSim)
            
            if Reviewer != None:
                Reviewer.ArticleAssigned.append(Article.Name)
                Article.ArticleAssignedTo.append(Reviewer.Name)
                selectedReviewer.append(Reviewer)
        
        elif SELECTED_MODE == 'MIXED_RB':
        
            Reviewer = SelectByRandomWalk (Article, ReviewersList, sim_thr,
                                       max_rvw, CalSim)

            if Reviewer != None:
                Reviewer.ArticleAssigned.append(Article.Name)
                Article.ArticleAssignedTo.append(Reviewer.Name)
                selectedReviewer.append(Reviewer)
             
            else:
                Reviewer = SelectByBruteForce (Article, ReviewersList, sim_thr,
                                           max_rvw, CalSim)
            
                if Reviewer != None:
                    Reviewer.ArticleAssigned.append(Article.Name)
                    Article.ArticleAssignedTo.append(Reviewer.Name)
                    selectedReviewer.append(Reviewer)

        elif SELECTED_MODE == 'MIXED_RR':
        
            Reviewer = SelectByRandomWalk (Article, ReviewersList, sim_thr,
                                       max_rvw, CalSim)

            if Reviewer != None:
                Reviewer.ArticleAssigned.append(Article.Name)
                Article.ArticleAssignedTo.append(Reviewer.Name)
                selectedReviewer.append(Reviewer)
             
            else:
                Reviewer = SelectByRandomBruteForce (Article, ReviewersList, 
                                                     sim_thr, max_rvw, CalSim)
            
                if Reviewer != None:
                    Reviewer.ArticleAssigned.append(Article.Name)
                    Article.ArticleAssignedTo.append(Reviewer.Name)
                    selectedReviewer.append(Reviewer)
        
        else:
            print ('Wrong Selection')

        count += 1

    return selectedReviewer

###########################################################
#               General Function
###########################################################

def ReviewersData (ArticlesList):
    ReviewersList = []
    
    full_path = [os.path.join(r,file) for r,d,f in os.walk(path3) for file in f]
    
    for i in range(0, len(full_path), 1):
        # Find the name of the reviewer
        Name = str(full_path[i].split('\\')[2]).split('.')[0]
        
        ReviewerPhrases = pd.read_csv (full_path[i])
        
        rwr = Reviewer (Name, ReviewerPhrases)
        ReviewersList.append(rwr)
    
    return ReviewersList

        
def FetchArticleDetails ():
    ArticlesList = []
    
    full_path = [os.path.join(r,file) for r,d,f in os.walk(path5) for file in f]
    
    for i in range(0, len(full_path), 1):
        # Find the name of the reviewer
        Name = str(full_path[i].split('\\')[2]).split('.')[0]
        
        Phrases = pd.read_csv (full_path[i])
        
        artc = Article (Name, Phrases)
        
        ArticlesList.append(artc)
    
    return ArticlesList



###########################################################
#                    Statistics                           #
###########################################################

def FindStatistics (ReviewersList, ArticlesList):

    fo = open(f'{path6}/ReviewStatistics.txt', 'a')
    
    reviewerStat = []
    
    for i in range (0, MAX_ASSIGNED_REVIEW + 1):
        reviewerStat.append(0)
        
        
    for Reviewer in ReviewersList:
        reviewerStat[Reviewer.NoOfArticleAssigned()] += 1
        
    fo.write(str(SELECTED_MODE) + ', ST: ' + str(SIMULATION_THRESHOLD) + ', RC: ' 
             + str(REVIEWER_COUNT) + ', MAR ' + str(MAX_ASSIGNED_REVIEW) + ', ')
    for i, item in enumerate(reviewerStat):
        if i < len(reviewerStat) - 1:
            fo.write(str(i) + ': ' + str(item) + ', ')
        else:
            fo.write(str(i) + ': ' + str(item))
    fo.write('\n')
    
    fo.close()

    fo = open(f'{path6}/ArticleStatistics.txt', 'a')
    
    articleStat = []
    
    for i in range (0, REVIEWER_COUNT + 1):
        articleStat.append(0)

    for Article in ArticlesList:
        articleStat[Article.NoOfReviewerAssigned()] += 1
        
    fo.write(str(SELECTED_MODE) + ', ST: ' + str(SIMULATION_THRESHOLD) + ', RC: ' 
             + str(REVIEWER_COUNT) + ', MAR ' + str(MAX_ASSIGNED_REVIEW) + ', ')
    for i, item in enumerate(articleStat):
        if i < len(articleStat) - 1:
            fo.write(str(i) + ': ' + str(item) + ', ')
        else:
            fo.write(str(i) + ': ' + str(item))
    fo.write('\n')
    
    fo.close()
    
###########################################################
#                   Print                                 #
###########################################################
    
def PrintResult (ReviewersList, ArticlesList):
    fo = open(f'{path6}/ReviewerSelection.txt', 'w')
    
    fo.write ('Name, Count, Articles\n')
    
    for Reviewer in ReviewersList:
        fo.write (Reviewer.Name + ', ' + str(Reviewer.NoOfArticleAssigned()) 
                  + ', ')
        for i, Article in enumerate(Reviewer.ArticleAssigned):
            if i < len(Reviewer.ArticleAssigned) - 1:
                fo.write (Article + ', ')
            else:
                fo.write(Article)
        fo.write('\n')
    
    fo.close()

    fo = open(f'{path6}/ArticleAssigned.txt', 'w')
    
    for Article in ArticlesList:
        fo.write (Article.Name + ', ' + str(Article.NoOfReviewerAssigned()) 
                  + ', ')
        for i, Reviewer in enumerate(Article.ArticleAssignedTo):
            if i < len(Article.ArticleAssignedTo) - 1:
                fo.write (Reviewer + ', ')
            else:
                fo.write(Reviewer)
        fo.write('\n')
    
    fo.close()

###########################################################
#               Main Function                             #
###########################################################

if __name__ == '__main__':
        
    ArticlesList = FetchArticleDetails()
    ReviewersList = ReviewersData(ArticlesList)
    
    for item in ArticlesList:
        SelectReviewers(item, ReviewersList, REVIEWER_COUNT,
                        SIMULATION_THRESHOLD, MAX_ASSIGNED_REVIEW)
    
    FindStatistics (ReviewersList, ArticlesList)
    PrintResult (ReviewersList, ArticlesList)
    
    
    '''
    print ('=========================================================')
    print ("Reviewer List")
    print ('=========================================================')
    print (ReviewersList)
    
    print ('=========================================================')
    print ("Article List")
    print ('=========================================================')
    print (ArticlesList)
    '''
    
    