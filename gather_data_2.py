import bz2
# import boto3
import json
import multiprocessing
import glob
import pandas_gbq as pg

## subreddits that are popular and/or text heavy
subreddit_lists = ['funny','worldnews','todayilearned','showerthoughts']


def bz2_reader(filepath_bz2, filepath_json):
    with bz2.BZ2File(filepath_bz2) as source_file, \
            open(filepath_json, 'w+') as output_file:
        for line in source_file:
            dic = json.loads(line)
            try:
                if dic['subreddit'] == 'AskReddit':
                    print(dic['subreddit'])
                    json.dump(dic, output_file)
                    output_file.write('\n')
            except:
                continue
                # line = str(line)





# s3 = boto3.resource('s3')

def upload_to_s3(file):
    data = open(file,'rb')
    s3.Bucket('fpolchow').put_object()


def join_files(file_name,outfile):
    read_files = glob.glob('c*.json')
    with open(output_filename,'w+') as output_filename:
        for f in read_files:
            with open(f,'rb') as infile:
                for line in source_file:
                        output_filename.writelines(line)


query = """WITH comments AS (
            SELECT body,CHAR_LENGTH(body) AS num_char, SUBSTR(link_id,4) AS link_id, id, score, created_utc,
            parent_id
            FROM `fh-bigquery.reddit_comments.2017_01`
            WHERE subreddit = "AskReddit")



            SELECT post.created_utc AS post_time, post.score AS post_score, post.id AS link_id,
            comments.num_char AS num_char, comments.id as comment_id, comments.score as score,
            comments.created_utc as comment_time, comments.body as text, comments.parent_id as parent_id
            FROM `fh-bigquery.reddit_posts.2017_01` AS post
            JOIN comments
            ON comments.link_id = post.id
            WHERE subreddit = "AskReddit"
            AND num_comments > 0;
            """

def import_data(query,project_id,output_table):
    data = pg.read_gbq(query, projectid,dialect='standard',
        configuration={
            'query': {
                'allowLargeResults': True,
                'destinationTable': {
                    'projectId': projectid,
                    'datasetId': 'redditcommentdata',
                    'tableId': output_table
                     }
                    }
                })
    data.to_csv('./data/'+output_table,)
    return data, output_table



def join_files(data,file_name):
    read_files = glob.glob('c*.json')
    with open(output_filename,'w+') as output_filename:
        for f in read_files:
            with open(f,'rb') as infile:
                for line in source_file:
                        output_filename.writelines(line)









if __name__ == '__main__':
    files = [('./data/RC_2016-02.bz2','./data/16-02_c.json'),('./data/RC_2016-03.bz2','./data/16-03_c.json'),\
            ('./data/RC_2016-04.bz2', './data/16-04_c.json')]

    p = multiprocessing.Pool(processes=3)

    bz2_reader('./data/RS_2017-04.bz2','./data/09_04_s.json')
    for i,o in files:
        # launch a process for each file
        # The result will be approximately one process per CPU core available.
        p.apply(bz2_reader, args = (i,o))

    p.close()
    p.join()