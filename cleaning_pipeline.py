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
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import datetime




def make_parent_dict(dataframe):
    """Creates a dictory of every comment and its score"""
    d = {}
    for index, row in dataframe.iterrows():
        #         print row

        d[row['comment_id']] = (row['score'],row['comment_time'])
    return d


def calculate_parent_score(x,args):
    """funtion to create new column that contains the parent score and the time of posting"""
    if x['parent_id'][:3] == 't3':
        return (x['post_score'], x['time_after_post'])
    elif x['parent_id'][3:] not in args:
        return x['post_score'], x['time_after_post']
    else:
        score,parent_time = args[x['parent_id'][3:]]
        time_diff = x['comment_time'] - parent_time
        return score, time_diff


#NLP Processing

def cleanup(comment):
    """ Eliminates punctuation and replaces the words"""
    cleanup_re = re.compile('[^a-zA-Z0-9]+')
    comment = comment.lower()
    regex_link = re.compile("(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z0-9][a-z0-9\-]*")
    comment = regex_link.sub('22link22', comment)
    comment = cleanup_re.sub(' ', comment).strip()
    return comment

with open('./data/swearWords.txt') as f:
    obscene = set([line.strip() for line in f.readlines()])

def cleanup_with_link_profanity(comment):

    """ Cleans punctuation in addition to counting the total number of links and 'profane' words as deemed by Carnegie
     Melon University"""

    cleanup_re = re.compile('[^a-z]+')
    comment = comment.lower()
    regex_link = re.compile("(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z0-9][a-z0-9\-]*")
    comment = regex_link.sub('linkpost', comment)
    comment = cleanup_re.sub(' ', comment).strip()
    word_list = nltk.word_tokenize(comment)
    profanity = 0
    link = 0
    cleaned_comment =''
    for word in word_list:
        if word in obscene:
            profanity += 1
        if word == 'linkinpost':
            link += 1
    cleaned_comment = " ".join(word_list)
    return cleaned_comment,profanity,link

porter = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))

def tokenizer(comment):
    """Tokenizes each comment"""
    doc = nltk.word_tokenize(comment)
    stemmed = [porter.stem(word) for word in doc if word not in stopwords]
    return stemmed



def time_difference(df):
    """Calculates the difference in time from original post to """
    df['time_after_post'] = df['comment_time'] - df['post_time']
    return df

def hour_of_day(row):
    """Determines the time of day a comment was posted"""
    return datetime.datetime.fromtimestamp(row['comment_time']).hour

def day_of_week(row):
    """Returns the day of the week something is posted"""
    return datetime.datetime.fromtimestamp(row['comment_time']).weekday()


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


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def add_sentiment_parallelize(df):
    df['pos_sentiment'] = df['cleaned'].apply(sentiment_analysis)
    return df

def add_link_profanity_parallelize(df):
    df['cleaned'],df['profanity'],df['link'] = zip(*df['body'].apply(cleanup_with_link_profanity))
    return df

def add_parent_score_and_time(df):
    df['parent_score'], df['time_after_parent'] = zip(*df.apply(calculate_parent_score, axis=1, dic=parent_dictionary))
    return df

def rmse(y_actual,y_predicted):
    return (mean_squared_error(y_actual,y_predicted))**0.5


def main():
    df = time_difference(df)
    parent_dictionary = make_parent_dict(df)
    df = add_parent_score_and_time(df)
    df = df.loc[(df['text'] != '[deleted]') & (df['text'] != '[removed]')]
    # df['cleaned'] = df['text'].apply(cleanup)
    ## use this if you want to include profanity in the mix

    # df['cleaned'], df['profanity'], df['link'] = comments['body'].apply(cleanup_with_link_profanity)
    df['time_of_day'] = df.apply(hour_of_day, axis=1)


    df['top'] = np.where(df['parent_id'].str.match("^t1_\S+", 1, 0))
    ##determine what level of
    well_liked(df,3)

    num_partitions = 50 #number of partitions to split dataframe
    num_cores = 20 #number of cores on your machine



    df = parallelize_dataframe(df, add_link_profanity_parallelize)

    tb = Blobber(analyzer=NaiveBayesAnalyzer())
    df = parallelize_dataframe(df,add_sentiment_parallelize)


    X = df[['cleaned','pos_sentiment','top','parent_score','time_of_day','time_after_post','num_char','profanity']]
    y = df[['liked','score']]


    X_train, X_test, y_train, y_test = test_train_split(X,y)

    liked_train = y_train['liked']
    liked_test = y_test['liked']
    score_train = y_train['score']
    score_test = y_test['score']
    text_train = X_train['cleaned']
    text_test = X_test['cleaned']

    ## we are done with the preprocessing, now it's time to make our predictive model that will
    ## feed into the ensemble

    ##generate the predictions
    ta = TextAnalysis(classifier=MultinomialNB(), method='tf_idf', tokenizer=tokenizer)
    ta.get_vectorizer(text_train,max_features=10000)
    ta.fit(text_train,liked_train)
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