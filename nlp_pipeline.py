import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import textblob
import nltk
from sklearn.model_selection import train_test_split
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import datetime
import argv
import sys


def make_parent_dict(dataframe):
    """Creates a dictory of every comment and its score"""
    d = {}
    for index, row in dataframe.iterrows():
        #         print row

        d[row['id']] = row['score']
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
stemmer = PorterStemmer()

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
    porter = nltk.stem.PorterStemmer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    doc = nltk.word_tokenize(comment)
    stemmed = [porter.stem(word) for word in doc if word not in stopwords]
    return stemmed



def time_difference(df):
    df['time_after_post'] = df['comment_time'] - df['post_time']


def hour_of_day(row):
    """Determines the time of day a comment was posted"""
    return datetime.datetime.fromtimestamp(row['created_utc']).hour



tb = Blobber(analyzer=NaiveBayesAnalyzer())
nltk.download()
def sentiment_analysis(comm):
    """Uses an sentiment prediction model pretrained on 100,000 movie reviews, the model takes care of tokenization"""

    classification, pos, neg = tb(comm).sentiment
    return  pos

def well_liked(df,num):
    df['liked'] = np.where(df['score']>num,1,0)


def freq_generator(train, test,
                   method='count',
                   tokenizer=None,
                   ngram_range=(1, 1),
                   max_features=30000, vocabulary=None):
    if method == 'count':
        vectorizer = CountVectorizer(tokenizer=tokenizer, \
                                     stop_words='english', \
                                     ngram_range=(1, 1),
                                     max_features=max_features)
        X_train = vectorizer.fit_transform(train)
        X_test = vectorizer.transform(test)
    elif method == 'tf_idf':
        vectorizer = TfidfVectorizer(tokenizer=tokenizer,
                                     ngram_range=ngram_range,
                                     max_features=max_features,
                                     vocabulary=vocabulary)
        X_train = vectorizer.fit_transform(train)
        X_test = vectorizer.transform(test)

    #     vdoc_term_matrix = vectorizer.fit_transform(x_train).toarray()

    return X_train, X_test


def get_vocabulary(train,
                   method='count',
                   tokenizer=None,
                   ngram_range=(1, 1),
                   max_features=30000, vocabulary=None):
    if method == 'count':
        vectorizer = CountVectorizer(tokenizer=tokenizer,
                                     stop_words='english',
                                     vocabulary=vocabulary,
                                     ngram_range=ngram_range)

        vectorizer.fit(train)
        vocab = vectorizer.vocabulary_.values

    elif method == 'tf_idf':
        vectorizer = TfidfVectorizer(tokenizer=tokenizer,
                                     ngram_range=ngram_range,
                                     max_features=max_features,
                                     vocabulary=vocabulary)
        vectorizer.fit(train)
        vocab = vectorizer.vocabulary_.values
    return vocab, vectorizer


def find_important_vocabulary(X_train,y_train,method,ngram_range,n_words,word_list):

    vocab,vect = freq_generator(X_train['cleaned'],method=method,ngram_range=ngram)
    rf = RandomForestClassifier()
    X_train = vect.transform(X_train['cleaned'])
    rf.fit(X_train,y_train)
    feature_importance = rf.feature_importances_
    word_list.extend(vocab[np.argsort(feature_importance)][:n_words])



def get_text_predictions(X_train,y_train, vect, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=5)
    X_train['text_classifier'] = 0
    X = X_train['cleaned'].values.astype('str')
    y = y_train['liked'].values

    score_list = []
    for train_index, test_index in kf.split(X):
        X_train_pred, X_test_pred = vect.fit_transform(X[train]),vect.transform(X[test])
        mnb = MultinomialNB()
        mnb.fit(X_train_pred, y[train_index])
        predictions = mnb.predict_proba(X_test_pred)
        X_train['text_classifier'].iloc[test_index] = predictions
        score_list.append(mnb.score(X_test, y[test]))



def main():
    # df = import_file()
    parent_dictionary = make_parent_dict(df)
    df['parent_score'] = df.apply(create_new_cols,axis = 1,dic = parent_dictionary)
    df = df.loc[(df['body'] != '[deleted]') & (df['body'] != '[removed]')]
    df['cleaned'] = df['body'].apply(cleanup)
    ## use this if you want to include profanity in the mix
    # df['cleaned'], df['profanity'], df['link'] = comments['body'].apply(cleanup_with_link_profanity)

    df['time_of_day'] = df.apply(hour_of_day, axis=1)
    time_difference(df)

    df['top'] = np.where(df['parent_id'].str.contains("c"), 1, 0)
    well_liked(df,3)

    df['pos_sentiment'] = df['cleaned'].apply(sentiment_analysis,axis=1)

    X = df[['cleaned','pos_sentiment','top','parent_score','time_of_day','time_after_post']]
    y = df['score']


    X_train, X_test, y_train, y_test = test_train_split(X,y)


    ## we are done with the preprocessing, now it's time to make our predictive model that will
    ## feed into the ensemble
    most_important_words = []
    ##finding most important unigrams
    find_important_vocabulary(X_train,method='tf_idf', ngram_range=(1,1), n_words=10000, word_list=most_important_words)

    ##finding most important bigrams
    find_important_vocabulary(X_train, method='tf_idf', ngram_range=(2, 2), n_words=1000, word_list=most_important_words)

    ##obtaining vectorizer for top unigrams and bigrams
    vocab, vect = get_vocabulary(X_train, method='tf_idf', ngram_range=(1, 2), vocabulary=most_important_words)

    ##generate the predictions
    get_text_predictions(X_train,y_train,)



if __name__ == '__main__':
    comments = import_data()
    submissions = import_data()comments = comments[['body','created_utc','edited','id',\
                                                'link_id','name','parent_id','score','subreddit']]
    comment_dict = make_parent_dict(merged)
    merged['parent_score'] = merged.apply(create_new_cols,axis=1,dic = comment_dict)


get-vocab
fit
get_training_prediction
