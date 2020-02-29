# -*- coding: utf-8 -*-


import nltk
#nltk.download()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

text =[
       'Each workday really depends on the length of the flight and my rotation.',

'These days I generally fly from my base in Atlanta to the West Coast, including Los Angeles, Portland, San Francisco, Las Vegas, and other destinations. After meeting with my fellow flight attendants on board, we perform the required safety and security checks before helping passengers board.',"We also prepare beverage carts and food carts for cabin service. If I'm working in the first-class cabin, I have meals to cook and work with pilots to discuss details of the flight.",

"I'm a people person, so after we finish our drink and snack service in the cabin I engage with customers and make sure I'm visible in the aisles to keep them well taken care of. We are there for our customers' safety and comfort, and every interaction counts, so I make the most of it when I'm in the aisles.",
"I like to see my passengers are happy"

       ]

sid = SentimentIntensityAnalyzer()
for sen in text:
    print(sen)
    
    ss = sid.polarity_scores(sen)
    for k in ss:
        print('{0}:{1},'.format(k,ss[k]),end='\n')