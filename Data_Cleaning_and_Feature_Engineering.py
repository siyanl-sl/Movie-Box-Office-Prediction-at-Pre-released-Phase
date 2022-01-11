#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
import json
import csv


#Data Preprocessing and Features Extraction


import pandas as pd
import json,ast


# ## 2.0 Data Cleaning



# read data
details=pd.read_csv('movie_details.csv')
revenue=pd.read_csv('domestic_revenue.csv')
revenue = revenue.rename(columns={"ttid": "imdb_id"})



# merge data
details = pd.merge(details, revenue, how = 'left', on=['imdb_id'])




# delete rows missing revenue
details = details[details['domestic_revenue'].notnull()]
details = details[details['budget']>=10000]


# ## 2.1 Inflation Adjustment 


inflation = {
    2000:1.484653988,
    2001:1.443848362,
    2002:1.421306,
    2003:1.389757192,
    2004:1.353520251,
    2005:1.309105611,
    2006:1.26819437,
    2007:1.233020338,
    2008:1.187433572,
    2009:1.191670512,
    2010:1.172441955,
    2011:1.136562478,
    2012:1.113519994,
    2013:1.097444272,
    2014:1.079925473,
    2015:1.078645906,
    2016:1.065207428,
    2017:1.042990581,
    2018:1.018122101,
    2019:1
}



def adj_budget (year, budget):
    new_budget = inflation[year]*budget
    return new_budget

# inflation on budget
movies = details.copy()
movies['new_budget'] = movies.apply(lambda x: adj_budget(x['year'], x['budget']), axis=1)

# inflation on revenue
movies1 = movies.copy()
movies1['new_rev'] = movies.apply(lambda x: adj_budget(x['year'], x['domestic_revenue']), axis=1)

# ## 2.2 Actor Starpower

credits = pd.read_csv('movie_casts_crew.csv')


import ast
popularity=[]
order=[]
for cast in credits['cast']:
    cast = ast.literal_eval(cast)
    pop=[]
    orde=[]
    top_3=cast[0:3]
    if len(top_3) > 0:
        for actor in top_3:
            pop.append(actor['popularity'])
            orde.append(actor['order'])
        pop_mean=np.mean(pop)
    else:
        pop_mean=np.nan
    popularity.append(pop_mean)




credits['actor_popularity']=popularity



# ## 2.3 Director Starpower



director_popularity=[]
for crews in credits['crew']:
    crews = ast.literal_eval(crews)
    directs=[]
    for crew in crews:
        if crew['job'] == 'Director':
            directs.append(crew)
    if len(directs)>0:
        pop_dir=[]
        for direct in directs:
            pop_dir.append(direct['popularity'])
        pop_dir_max = np.max(pop_dir)
    else:
        pop_dir_max = np.nan
    director_popularity.append(pop_dir_max)


credits['director_popularity']=director_popularity
credits = credits[['id','director_popularity','actor_popularity']]
credits


# ## 2.4 Sequel



movies1['sequel']=movies1['belongs_to_collection'].apply(lambda x: 'No' if pd.isna(x) else 'Yes')




# ## 2.5 Language


def language_transformation(language_list):
    new_lanlist = []
    for lan in language_list:
        if lan=="en":
            new_lanlist.append('en')
        else:
            new_lanlist.append('others')
    return new_lanlist


new_lan = language_transformation(list(movies1['original_language']))
movies1['new_language'] = new_lan


# ## 2.6 Genre


# nineteen official genres according to TMDB API
gen={"genres":[{"id":28,"name":"Action"},{"id":12,"name":"Adventure"},{"id":16,"name":"Animation"},{"id":35,"name":"Comedy"},{"id":80,"name":"Crime"},{"id":99,"name":"Documentary"},{"id":18,"name":"Drama"},{"id":10751,"name":"Family"},{"id":14,"name":"Fantasy"},{"id":36,"name":"History"},{"id":27,"name":"Horror"},{"id":10402,"name":"Music"},{"id":9648,"name":"Mystery"},{"id":10749,"name":"Romance"},{"id":878,"name":"Science Fiction"},{"id":10770,"name":"TV Movie"},{"id":53,"name":"Thriller"},{"id":10752,"name":"War"},{"id":37,"name":"Western"}]}


# get genre list
gen_names=[]
for i in range(0,19):
    gen_name=gen['genres'][i]['name']
    gen_names.append(gen_name)


# Make each genre a column
for gen_name in gen_names:
    movies1 = pd.concat([movies1, pd.DataFrame(columns=[gen_name])], sort=False)

movies2 = movies1.copy()
for idx in movies2.index:
    genre_i=movies2['genres'][idx]
    genre_i_dict = ast.literal_eval(genre_i)#change str to list
    for i in range(len(genre_i_dict)):
        genre_i_name=genre_i_dict[i]['name']       
        for j in range(len(gen_names)):
            if genre_i_name==gen_names[j]:
                movies2[gen_names[j]][idx]=1


for genre in gen_names:
    movies2[genre] = movies2[genre].fillna(0)
movies2


# ## 2.7 Company features: company size and counts



# prepare for the company features 
production_companies = list(movies2['production_companies'])



def count_frequency(company_list):
    id_count = {}
    for movie_companies in company_list:
        jsontmp = eval(movie_companies)
        for json_dict in jsontmp:
            id = json_dict['id']
            id_count[id] = 1 if id not in id_count else id_count[id]+1

    return id_count



from sklearn.cluster import KMeans
# use K-means to classify company class
def create_company_class(company_count,threasholds,n_clusters):
    classes = {}
    x=np.array(list(company_count.values())).reshape(-1,1)
    kmeans = KMeans(n_clusters=n_clusters).fit(x)
    k_classes = kmeans.labels_

    for company in list(company_count.keys()):
        count = company_count[company]
        if count<=threasholds[0]:
            classes[company] = 'small'
        elif count<=threasholds[1]:
            classes[company] = 'medium'
        else:
            classes[company] = 'big'

    return classes



def find_class_count(company_list,class_dict):
    company_counts = []
    max_classes = []
    for movie_companies in company_list:
        jsontmp = eval(movie_companies)
        company_counts.append(len(jsontmp))
        maxclass = 'small'
        for json_dict in jsontmp:
            classtmp = class_dict[json_dict['id']]
            if classtmp=='medium' and maxclass == 'small':
                maxclass = 'medium'
            
            if classtmp=='big':
                maxclass='big'
        max_classes.append(maxclass)
    return company_counts,max_classes


test = count_frequency(production_companies)



n_clusters = 10
threasholds = [10,100]
test_class = create_company_class(test,threasholds,n_clusters)
company_counts, company_classes = find_class_count(production_companies,test_class) 


movies2['company_count'] = company_counts
movies2['company_class'] = company_classes


# ## 2.8 Release date features: month and holiday


from datetime import datetime


# prepare for the holiday features

holiday_duration = 21
holiday_dict = {'Christmas':{'start':'12/18','end':'01/07'}}

def judge_holiday(datetmp):
    for holiday in holiday_dict:
        if datetmp.month<2:
            end = str(datetmp.year)+'/'+holiday_dict[holiday]['end']
            end_date = datetime.strptime(end,'%Y/%m/%d').date()
            if datetmp<end_date:
                return 'yes'
        else:
            start = str(datetmp.year)+'/'+holiday_dict[holiday]['start']
            start_date = datetime.strptime(start,'%Y/%m/%d').date()
            if datetmp>start_date:
                return 'yes'
        
    return 'no'



def date_to_month_hol(datelist):
    months = []
    holiday = []
    year = []
    for datestr in datelist:
        date1 = datetime.strptime(datestr,'%Y/%m/%d').date()
        months.append(str(date1.month))
        year.append(str(date1.year))
        holiday.append(judge_holiday(date1))

    return months,holiday,year



months, holidays, year = date_to_month_hol(list(movies2['release_date']))




movies2['month'] = months
movies2['holiday'] = months


# ## 2.9 Conduct text feature mining using word2vec and use PCA

# text mining and pca feature
import re
import operator
import argparse
import codecs
import os
import gensim



 
def isNumber(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False
 

inputdir = os.getcwd()
print(inputdir)
class Rake:
    
    def __init__(self, text_dataset, stopwordsFilePath, outputFilePath, minPhraseChar, maxPhraseLength):
        self.outputFilePath = outputFilePath
        self.minPhraseChar = minPhraseChar
        self.maxPhraseLength = maxPhraseLength
        # read documents
        self.docs = []
        # for document in codecs.open(inputFilePath, 'r', 'utf-8'):
        #     self.docs.append(document)
        for document in text_dataset:
            self.docs.append(document)
        # read stopwords
        stopwords = []
        for word in codecs.open(stopwordsFilePath, 'r', 'utf-8'):
            stopwords.append(word.strip())
        stopwordsRegex = []
        for word in stopwords:
            regex = r'\b' + word + r'(?![\w-])'
            stopwordsRegex.append(regex)
        self.stopwordsPattern = re.compile('|'.join(stopwordsRegex), re.IGNORECASE)
 
    def separateWords(self, text):
        splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        words = []
        for word in splitter.split(text):
            word = word.strip().lower()
            # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if len(word) > 0 and word != '' and not isNumber(word):
                words.append(word)
        return words
    
    
    def calculatePhraseScore(self, phrases):
        # calculate wordFrequency and wordDegree
        wordFrequency = {}
        wordDegree = {}
        for phrase in phrases:
            wordList = self.separateWords(phrase)
            wordListLength = len(wordList)
            wordListDegree = wordListLength - 1
            for word in wordList:
                wordFrequency.setdefault(word, 0)
                wordFrequency[word] += 1
                wordDegree.setdefault(word, 0)
                wordDegree[word] += wordListDegree
        for item in wordFrequency:
            wordDegree[item] = wordDegree[item] + wordFrequency[item]
    
        # calculate wordScore = wordDegree(w)/wordFrequency(w)
        wordScore = {}
        for item in wordFrequency:
            wordScore.setdefault(item, 0)
            wordScore[item] = wordDegree[item] * 1.0 / wordFrequency[item]
 
        # calculate phraseScore
        phraseScore = {}
        for phrase in phrases:
            phraseScore.setdefault(phrase, 0)
            wordList = self.separateWords(phrase)
            candidateScore = 0
            for word in wordList:
                candidateScore += wordScore[word]
            phraseScore[phrase] = candidateScore
        return phraseScore
    
        
    def execute(self):
        all_text = []
        all_score = []
        for document in self.docs:
            # split a document into sentences
            sentenceDelimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
            sentences = sentenceDelimiters.split(document)
            # generate all valid phrases
            phrases = []
            for s in sentences:
                tmp = re.sub(self.stopwordsPattern, '|', s.strip())
                phrasesOfSentence = tmp.split("|")
                for phrase in phrasesOfSentence:
                    phrase = phrase.strip().lower().split()
                    
                    if phrase != "" and len(phrase) >= self.minPhraseChar and len(phrase) <= self.maxPhraseLength:
                        phrases+=phrase
            all_text.append(phrases)        

        model1 = gensim.models.Word2Vec(all_text, min_count = 1, 
                              window = 5)
        print(len(model1.wv['quick']))
        for textline in all_text:
            score = np.zeros(100)
            score_tmp = []
            for words in textline:
                score+=model1.wv[words]
            
            for i in range(10):
                sctmp = 0
                for j in range(i*10,(i+1)*10):
                    sctmp += score[j]
                score_tmp.append(sctmp)
            all_score.append(score_tmp)

        all_score_df = pd.DataFrame(all_score)
        return all_score_df

movies_origin = movies2.copy()
new_text = []
for i,texttmp in enumerate(list(movies_origin['overview'])):
    tmp = texttmp
    tmp += list(movies_origin['title'])[i]
    tmp += str(list(movies_origin['tagline'])[i])
    new_text.append(tmp)
movies_origin['new_textline'] = new_text
rake = Rake(list(movies_origin['new_textline']), inputdir+'\\stop_words.txt', inputdir+'\\result_Quantum.csv', 1, 100)

text_df = rake.execute()

from sklearn.decomposition import PCA

pca = PCA(n_components=1)  
pca.fit(text_df.values)                 
newX=pca.fit_transform(text_df.values)   
print(pca.explained_variance_ratio_)
new_text_df = pd.DataFrame(newX)

new_text_df['imdb_id'] = movies_origin['imdb_id']
new_text_df['id'] = movies_origin['id']
movies_origin = movies_origin.merge(new_text_df,how='left')


