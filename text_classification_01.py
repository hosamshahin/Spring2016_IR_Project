from __future__ import print_function
import globals
from globals import load_config
import codecs, re, json, os, time
from pyspark import SparkContext, SparkConf
from pyspark.mllib.fpm import FPGrowth
from pyspark.sql import SQLContext, Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, IDF
# , StopWordsRemover
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


if __name__ == "__main__":
    sc = SparkContext(appName="BinaryClassificationMetricsExample")
    sqlContext = SQLContext(sc)


    def parse_tweet(line):
        """
        Parses a tweet record having the following format collectionId-tweetId<\t>tweetString
        """
        fields = line.strip().split("\t")
        if len(fields) == 2:
            # The following regex just strips of an URL (not just http), any punctuations,
            # or Any non alphanumeric characters
            # http://goo.gl/J8ZxDT
            text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",fields[1]).strip()
            # remove terms <= 2 characters
            text = ' '.join(filter(lambda x: len(x) > 2, text.split(" ")))
            # return tuple of (collectionId-tweetId, text)
            return (fields[0], text)


    def Load_tweets(collection_id):
        tweets_file = os.path.join(globals.data_dir , "z_" + collection_id)
        print("Loading " + tweets_file)
        if not os.path.isdir(tweets_file):
            print(tweets_file + " folder doesn't exist.")
            return False
        tweets = sc.textFile(tweets_file) \
                  .map(parse_tweet) \
                  .filter(lambda x: x is not None) \
                  .map(lambda x: Row(id=x[0], text=x[1])) \
                  .toDF() \
                  .cache()
        return tweets


    def preprocess_tweets(tweets):
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        tweets = tokenizer.transform(tweets)
        # remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        # tweets = remover.transform(tweets)
        return tweets


    # Frequent pattern mining expect each row to have a unique set of tokens
    def save_unique_token(tweets):
        tweets = (tweets
          .rdd
          .map(lambda x : (x.id, x.text, list(set(filter(None, x.filtered)))))
          .toDF()
          .withColumnRenamed("_1","id")
          .withColumnRenamed("_2","text")
          .withColumnRenamed("_3","filtered")).cache()
        return tweets


    def run_FPM(tweets, collection):
        model = FPGrowth.train(tweets.select("filtered").rdd.map(lambda x: x[0]), minSupport=0.02)
        result = sorted(model.freqItemsets().collect(), reverse=True)
        # sort the result in reverse order
        sorted_result = sorted(result, key=lambda item: int(item.freq), reverse=True)

        # save output to file
        with codecs.open(FP_dir + time.strftime("%Y%m%d-%H%M%S") + '_'
                                + collection["Id"] + '_'
                                + collection["name"] + '.txt', 'w',encoding='utf-8') as file:
            for item in sorted_result:
                file.write("%s %s\n" % (item.freq, ' '.join(item.items)))


    config_data = load_config(globals.config_file)
    for x in config_data["collections"]:
        tweets = Load_tweets(x["Id"])
        if tweets:
            tweets = preprocess_tweets(tweets)
            tweets = save_unique_token(tweets)
            run_FPM(tweets, x)