from __future__ import print_function
import globals
from globals import load_config
import codecs, re, json, os, time
from pyspark import SparkContext, SparkConf
from pyspark.mllib.fpm import FPGrowth
from pyspark.sql import SQLContext, Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, StopWordsRemover
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


if __name__ == "__main__":
    sc = SparkContext(appName="BinaryClassificationMetricsExample")
    sqlContext = SQLContext(sc)


    def parse_tweet(line):
        """
        Parses a tweet record having the following format collectionId-tweetId<\t>tweetString
        
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
        """
        # The following regex just strips of an URL (not just http), any punctuations,
        # or Any non alphanumeric characters
        # http://goo.gl/J8ZxDT
        text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",json.loads(line[1])["value"]).strip()
        # remove terms <= 2 characters
        text = ' '.join(filter(lambda x: len(x) > 2, text.split(" ")))

        return (line[0], text)


    def Load_tweets(collection_id):
        
        # Other options for configuring scan behavior are available. More information available at
        # https://github.com/apache/hbase/blob/master/hbase-server/src/main/java/org/apache/hadoop/hbase/mapreduce/TableInputFormat.java
        inputConf = {"hbase.zookeeper.quorum": globals.host, "hbase.mapreduce.inputtable": globals.table}
        keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"
        valueConv = "org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter"

        hbase_rdd = sc.newAPIHadoopRDD(
        "org.apache.hadoop.hbase.mapreduce.TableInputFormat",
        "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
        "org.apache.hadoop.hbase.client.Result",
        keyConverter=keyConv,
        valueConverter=valueConv,
        conf=inputConf)

        broadcastCollectionNumber = sc.broadcast(str(collection_id))

        print("Loading " + collection_id)

        tweets = hbase_rdd.flatMapValues(lambda v: v.split("\n")) \
                  .filter(lambda x: x[0].startswith(broadcastCollectionNumber.value) \
                    and json.loads(x[1])["columnFamily"] == "collection-management") \
                  .map(parse_tweet) \
                  .map(lambda x: Row(id=x[0], text=x[1])) \
                  .toDF() \
                  .cache()
        return tweets


    def preprocess_tweets(tweets):
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        tweets = tokenizer.transform(tweets)
        remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        tweets = remover.transform(tweets)
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
        with codecs.open(globals.FP_dir + "/" + time.strftime("%Y%m%d-%H%M%S") + '_'
                                + collection["Id"] + '_'
                                + collection["name"] + '.txt', 'w',encoding='utf-8') as file:
            for item in sorted_result:
                file.write("%s %s\n" % (item.freq, ' '.join(item.items)))


    config_data = load_config(globals.config_file)

    for x in config_data["collections"]:
        print("GOING TO GET " + x["Id"])
        tweets = Load_tweets(str(x["Id"]))
        print("Collected " + x["Id"])

        if tweets:
            tweets = preprocess_tweets(tweets)
            tweets = save_unique_token(tweets)
            run_FPM(tweets, x)
    