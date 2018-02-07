## Project Proposal

### Reddit Karma Predictor

**Description**: The main goal of my project is to use features of reddit comments to determine what goes into one that is highly-rated users. I will analyze this overall and on a subreddit basis. In the process, I plan on gathering a lot of different exploratory analysis on each subreddit as well as gain insight from how everything has been affected by time.

**Problem**: People put a lot of pride into how many upvotes they receive from comments on websites. It keeps people engaged on Reddit and make people feel validated for their opinions. A problem that Reddit might have is creating an algorithm to ensure that people are rewarded for the quality of their comments, rather than simply when they have posted. Better quality posts lead to a better quality community. If reddit can recognize these comments as “quality,” then it can promote them.

**Tools to use**: Because the data is so large (over a TB), I will have to use an AWS instance and Spark to manipulate the data. I plan on conducting NLP analysis on the comments with Tf -Idf vectorizer. To determine what the most important features are, I can use feature reduction techniques. 

**Deliverables**: I plan on having two different deliverables. One will be a web app, that will give someone a range of scores that they will likely get for a given comment in a given subreddit (most likely limited to the current top 10 subreddits). I’m breaking it up by subreddit because I think that each subreddit has a unique culture and there are different reasons why a comment might qualify as “upvote worthy.”
I also plan on providing a visualization of the similarity of subreddits to one another. This can be done with clustering or cosine similarity.
I also want to perform sentiment analysis to determine the overall sentiment from each subreddit.

**Data**: The data is found at http://files.pushshift.io/reddit/comments/ in .bz2 files, which are compressed text files.

**Next Steps**:
1. Determine how to merge the datasets together
2. Load into a dataframe (I don’t exactly know how to do this given the size of all the data)
3. Perform EDA on the data to determine some general trends (maybe get a word cloud of what words are used on reddit, how reddit has grown over time, what the most popular reddit groups are)
