# -*- coding: utf-8 -*-
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import xlrd
import matplotlib.pyplot as plt

pos_score=[]
neu_score=[]
neg_score=[]
compound_score=[]

def sentiment (path,page_no):
    #######################################Text Analytics
    text = xlrd.open_workbook(path)
    table = text.sheets()[page_no]  #打开excel第几个sheet
    nrows = table.nrows    #获得行数
    ncols = table.ncols    #获得列数

    for i in range(0,nrows):    #在所有行内
        rowValues = table.row_values(i)  #获得内容
    
        for item in rowValues:
            content = nltk.sent_tokenize(item) #nltk分句
            #print(content)
            
    SIA = SentimentIntensityAnalyzer()
    for sentence in content:
        print(sentence)
    
        ss = SIA.polarity_scores(sentence)
        pos_score.append(ss["pos"])
        neu_score.append(ss["neu"])
        neg_score.append(ss["neg"])
        compound_score.append(ss["compound"])
      
        for k in ss:
            print('{0}:{1},'.format(k,ss[k]),end='\n')
      #k是neu，pos，neg 等参数,所以ss[k]就为该参数得分
      
############################################### visualisation
    number = np.arange(1,len(pos_score)+1,1)
    sentence_number= []
    for n in number:
        a="Sentence %s"%n
        sentence_number.append(a)
        
    plt.xlabel('Number of Sentence')
    plt.ylabel("Probability")
    plt.plot(sentence_number,pos_score,label = 'positive feeling',color='g')
    plt.plot(sentence_number,neg_score,label="negative feeling",color='r')
    plt.plot(sentence_number,neu_score,label='neutral feeling',color='b')
    plt.plot(sentence_number,compound_score,label= 'complexity',color='orange')
    plt.legend()
    plt.show()
      
      
sentiment("C:\\Users\\laoyu\\Desktop\\编程软件\\Python\\NLP\\情感分析\\delta.xls",0)
