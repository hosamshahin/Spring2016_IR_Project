from __future__ import print_function
import globals
from globals import load_config
import codecs, re, json, os, time, sys
from pyspark import SparkContext, SparkConf
from pyspark.mllib.fpm import FPGrowth
from pyspark.sql import SQLContext, Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, StopWordsRemover
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#========================
# text_classification_01
#========================
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
              .filter(lambda x: x[0].startswith(broadcastCollectionNumber.value) and json.loads(x[1])["columnFamily"] == "collection-management") \
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

#========================
# text_classification_02
#========================
def create_training_data(tweets, freq_patterns):
    # Tweets contains the frequent pattern terms will be considered as positive samples
    positive_tweets = (tweets
      .rdd
      .filter(lambda x: set(freq_patterns).issubset(x.filtered))
      .map(lambda x : (x[0], x[1], x[2], 1.0))
      .toDF()
      .withColumnRenamed("_1","id")
      .withColumnRenamed("_2","text")
      .withColumnRenamed("_3","filtered")
      .withColumnRenamed("_4","label"))

    # calculate a fraction of positive samples to extract equivalent number of negative samples
    positive_fraction = float(positive_tweets.count()) / tweets.count()

    # Negative samples will be randomly selected from non_positive samples
    negative_tweets = (tweets
      .rdd
      .filter(lambda x: not set(freq_patterns).issubset(x[2]))
      .sample(False, positive_fraction, 12345)
      .map(lambda x : (x[0], x[1], x[2], 0.0))
      .toDF()
      .withColumnRenamed("_1","id")
      .withColumnRenamed("_2","text")
      .withColumnRenamed("_3","filtered")
      .withColumnRenamed("_4","label"))
    training_data = positive_tweets.unionAll(negative_tweets)
    return training_data


def train_lg(training_data, collection):
    # Configure an ML pipeline, which consists of the following stages: hashingTF, idf, and lr.
    hashingTF = HashingTF(inputCol="filtered", outputCol="TF_features")
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
    pipeline1 = Pipeline(stages=[hashingTF, idf])

    # Fit the pipeline1 to training documents.
    model1 = pipeline1.fit(training_data)

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    pipeline2 = Pipeline(stages=[model1, lr])

    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [10, 100, 1000, 10000]) \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .build()

    crossval = CrossValidator(estimator=pipeline2,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=5)

    # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(training_data)

#     model_path = os.path.join(models_dir , time.strftime("%Y%m%d-%H%M%S") + '_'
#                             + collection["Id"] + '_'
#                             + collection["name"])
#     cvModel.save(sc, model_path)
    return cvModel

def get_training_score_lg(lg_model, training_data):
    training_prediction = lg_model.transform(training_data)
    selected = training_prediction.select("label", "prediction").rdd.map(lambda x: (x[0], x[1]))
    training_error = selected.filter(lambda (label, prediction): label != prediction).count() / float(tweets.count())
    print("Training Error = " + str(training_error))


def create_testing_data(tweets):
    testing_data = (tweets
                    .rdd
                    .map(lambda x: Row(id=x[0], filtered=x[2]))
                    .toDF())
    return testing_data


def lg_prediction(lg_model, testing_data, collection):
    # Perfom predictions on test documents and save columns of interest to a file.
    prediction = lg_model.transform(testing_data)
    selected = prediction.select("id", "prediction", "probability")
    prediction_path = os.path.join(globals.predictions_dir , time.strftime("%Y%m%d-%H%M%S") + '_'
                            + collection["Id"] + '_'
                            + collection["name"])
    print(prediction_path)

    write_records_to_hbase(selected.collect())
    #def saveData(data):
    #    with open(prediction_path, 'a') as f:
    #        f.write(data.id+"\t"+str(data.probability[1])+"\n")
    #selected.foreach(saveData)

def write_records_to_hbase(records):
    outputConf = {"hbase.zookeeper.quorum": globals.host,
        "hbase.mapred.outputtable": globals.table,
        "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat",
        "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
        "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}
    keyConv = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"

    def mapDataset(x):
    	# print(x): Row(id=u'541-553091631943917568', prediction=1.0, probability=DenseVector([0.0056, 0.9944]))
        rowkey = str(x.id)
        columnFamily = "classification"
        columnName = "probability2"
        value = str(x.probability[1])
        return (rowkey, [rowkey, columnFamily, columnName, value])

    # ( rowkey , [ row key , column family , column name , value ] )
    rdd = sc.parallelize(records).map(mapDataset).saveAsNewAPIHadoopDataset(
    #records.map(mapDataset).saveAsNewAPIHadoopDataset(
      conf=outputConf,
      keyConverter=keyConv,
      valueConverter=valueConv)


if __name__ == "__main__":
    # set up spark context
    sc = SparkContext(appName="BinaryClassificationMetricsExample")
    sqlContext = SQLContext(sc)

    # get FP from config file
    config_data = load_config(globals.config_file)

    if len(sys.argv) < 1:
        print("Please specify 'FPM' or 'predict'.")
        exit(-1)

    if sys.argv[1] == "FPM":
        for x in config_data["collections"]:
            tweets = Load_tweets(str(x["Id"]))
            print("Collected " + x["Id"])

            if tweets:
                tweets = preprocess_tweets(tweets)
                tweets = save_unique_token(tweets)
                run_FPM(tweets, x)
    elif sys.argv[1] == "predict":
        for x in config_data["collections"]:
            tweets = Load_tweets(str(x["Id"]))
            if tweets:
                freq_patterns = x["FP"]
                tweets = preprocess_tweets(tweets)
                training_data = create_training_data(tweets, freq_patterns)
                lg_model = train_lg(training_data, x)
                get_training_score_lg(lg_model, training_data)
                testing_data = create_testing_data(tweets)
                lg_prediction(lg_model, testing_data, x)
    else:
        print("Please specify 'FPM' or 'predict'.")
        exit(-1)