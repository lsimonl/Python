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
    table = text.sheets()[page_no]  
    nrows = table.nrows    
    ncols = table.ncols    

    for i in range(0,nrows):    
        rowValues = table.row_values(i)  
    
        for item in rowValues:
            content = nltk.sent_tokenize(item)
            
            
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
            
     
      
############################################### visualisation
    number = np.arange(1,len(pos_score)+1,1)
    sentence_number= []
    for n in number:
        a="Sentence %s"%n
        sentence_number.append(a)
    
    plt.figure()
    plt.title("Sentiment Analyse")
    plt.xlabel('Number of Sentence')
    plt.ylabel("Probability")
    plt.plot(sentence_number,pos_score,label = 'positive feeling',color='g')
    plt.plot(sentence_number,neg_score,label="negative feeling",color='r')
    plt.legend()
    plt.show()
      
      
sentiment("C:\\Users\\laoyu\\Desktop\\Work, Society and organisation\\Summative2\\callcentre.xls",0)
sentiment("C:\\Users\\laoyu\\Desktop\\Work, Society and organisation\\Summative2\\callcentre.xls",1)
