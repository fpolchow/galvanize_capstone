import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import textblob
import nltk
from sklearn.model_selection import train_test_split
# from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import datetime
from sys import argv



def make_parent_dict(dataframe):
    """Creates a dictory of every comment and its score"""
    d = {}
    for index, row in dataframe.iterrows():
        #         print row

        d[row['comment_id']] = row['score']
    return d


def calculate_parent_score(x,**kwargs):
    """funtion to create new column that contains the parent score and the time of posting"""
    if x['parent_id'][:3] == 't3':
        return x['post_score']
    elif x['parent_id'][3:] not in kwargs:
        return x['post_score']
    else:
        score = kwargs[x['parent_id'][3:]]
        return score


#NLP Processing

def cleanup(comment):
    """ Eliminates punctuation and replaces the words"""
    cleanup_re = re.compile('[^a-zA-Z0-9]+')
    comment = comment.lower()
    regex_link = re.compile("(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z0-9][a-z0-9\-]*")
    comment = regex_link.sub('22link22', comment)
    comment = cleanup_re.sub(' ', comment).strip()
    return comment

def cleanup_with_link_profanity(comment):

    """ Cleans punctuation in addition to counting the total number of links and 'profane' words as deemed by Carnegie
     Melon University"""

    cleanup_re = re.compile('[^a-z0-9]+')
    comment = comment.lower()
    regex_link = re.compile("(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z0-9][a-z0-9\-]*")
    comment = regex_link.sub('22link22', comment)
    comment = cleanup_re.sub(' ', comment).strip()
    word_list = nltk.word_tokenize(comment)
    profanity = 0
    link = 0
    cleaned_comment =''
    for word in word_list:
        if word in profanity_list:
            profanity += 1
        if word == '22link22':
            link += 1
        cleaned_comment = " ".join(word_list)
    return cleaned_comment,profanity,link,


def tokenizer(comment):
    """Tokenizes each comment"""
    porter = nltk.stem.PorterStemmer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    doc = nltk.word_tokenize(comment)
    stemmed = [porter.stem(word) for word in doc if word not in stopwords]
    return stemmed



def time_difference(df):
    """Calculates the difference in time from original post to """
    df['time_after_post'] = df['comment_time'] - df['post_time']


def hour_of_day(row):
    """Determines the time of day a comment was posted"""
    return datetime.datetime.fromtimestamp(row['comment_time']).hour

def day_of_week(row):
    """Returns the day of the week something is posted"""
    return datetime.datetime.fromtimestamp(row['comment_time']).weekday()

tb = Blobber(analyzer=NaiveBayesAnalyzer())
# nltk.download()
def sentiment_analysis(comm):
    """Uses an sentiment prediction model pretrained on 100,000 movie reviews, the model takes care of tokenization"""

    classification, pos, neg = tb(comm).sentiment
    return  pos
# Add multiprocessing for this the computationally expensive process of calculating the sentiment for each comment

def well_liked(df,num):
    """Creates the binary classifier for whether or not a post is 'well-liked' by the community. This is up to the discretion
        of the data scientist. I have tried to ensure that the post is maintained."""
    df['liked'] = np.where(df['score']>num,1,0)
num_partitions = 50 #number of partitions to split dataframe
num_cores = 16 #number of cores on your machine

def multiply_columns(data):
    data['length_of_word'] = data['species'].apply(lambda x: len(x))
    return data

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def add_sentiment_parallelize(data):
    df['pos_sentiment'] = df['cleaned'].apply(sentiment_analysis)
    return df

df = parallelize_dataframe(df,add_sentiment_parallelize)





def main():
    # df = import_file()
    parent_dictionary = make_parent_dict(df)
    df['parent_score'] = df.apply(calculate_parent_score,axis = 1,dic = parent_dictionary)
    df = df.loc[(df['text'] != '[deleted]') & (df['text'] != '[removed]')]
    df['cleaned'] = df['text'].apply(cleanup)
    ## use this if you want to include profanity in the mix
    # df['cleaned'], df['profanity'], df['link'] = comments['body'].apply(cleanup_with_link_profanity)

    df['time_of_day'] = df.apply(hour_of_day, axis=1)
    time_difference(df)

    df['top'] = np.where(df['parent_id'].str.match("^t1_\S+", 1, 0))
    ##determine what level of
    well_liked(df,3)

    # num_partitions = 50 #number of partitions to split dataframe
    # num_cores = 16 #number of cores on your machine

    # def multiply_columns(data):
    #     data['length_of_word'] = data['species'].apply(lambda x: len(x))
    #     return data

    # def parallelize_dataframe(df, func):
    #     df_split = np.array_split(df, num_partitions)
    #     pool = Pool(num_cores)
    #     df = pd.concat(pool.map(func, df_split))
    #     pool.close()
    #     pool.join()
    #     return df

    # def add_sentiment_parallelize(data):
    #     df['pos_sentiment'] = df['cleaned'].apply(sentiment_analysis)
    #     return df

    # df = parallelize_dataframe(df,add_sentiment_parallelize)


    df['pos_sentiment'] = df['cleaned'].apply(sentiment_analysis)

    X = df[['cleaned','pos_sentiment','top','parent_score','time_of_day','time_after_post','num_char']]
    y = df['score']


    X_train, X_test, y_train, y_test = test_train_split(X,y)


    ## we are done with the preprocessing, now it's time to make our predictive model that will
    ## feed into the ensemble
    # most_important_words = []
    # ##finding most important unigrams
    # find_important_vocabulary(X_train,method='tf_idf', ngram_range=(1,1), n_words=10000, word_list=most_important_words)
    #
    # ##finding most important bigrams
    # find_important_vocabulary(X_train, method='tf_idf', ngram_range=(2, 2), n_words=1000, word_list=most_important_words)
    #
    # ##obtaining vectorizer for top unigrams and bigrams
    # vocab, vect = get_vocabulary(X_train, method='tf_idf', ngram_range=(1, 2), vocabulary=most_important_words)

    ##generate the predictions
    # get_text_predictions(X_train,y_train,)
    text_train = X_train['cleaned']
    ta = TextAnalysis(classifier=MultinomialNB, method='tf_idf',n_kfolds =5,tokenizer=tokenizer)
    ta.get_vocabulary(text_train,y_train,ngram = (1,2),n_words=20000)
    ta.train_predictions(text_train,y_train)
    rem = RedditEnsembleModel()
    rem.fit(X_train,y_train)


#
# if __name__ == '__main__':
#     main()
#     comments = import_data()
#     submissions = import_data()comments = comments[['body','created_utc','edited','id',\
#                                                 'link_id','name','parent_id','score','subreddit']]
#     comment_dict = make_parent_dict(merged)
#     merged['parent_score'] = merged.apply(create_new_cols,axis=1,dic = comment_dict)
main(df)