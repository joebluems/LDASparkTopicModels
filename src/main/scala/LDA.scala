package com.mapr.topics 

import scala.collection.mutable
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Main extends App { 
  Logger.getLogger("org").setLevel(Level.WARN)
  var corpusFile = "./documents"
  var k = 10  // number of topics
  var outputFile = "./topicsOut"
  val vocabSize = 1000 // total vocabulary size
  val iter = 10 // iterations
    
  if (args.length == 3) {
      corpusFile = args(0)
      k = args(1).toInt
      outputFile = args(2)
  }

  val spark = SparkSession.builder.master("local").appName("example").getOrCreate()
  import spark.implicits._

  def buildDataPrepPipeline(dataset: DataFrame, vocabSize: Int): (PipelineModel, Array[String]) = {
    val countTokens = udf { (words: Seq[String]) => words.length }
    val stop = StopWordsRemover.loadDefaultStopWords("english") ++
        Array("tr", "td", "div", "class", "table", "html", "div")
    val tokenizer = new RegexTokenizer().setInputCol("doc").setOutputCol("rawwords")
        .setGaps(false).setPattern("[a-zA-Z]{3,}")
    val stopremover = new StopWordsRemover().setInputCol("rawwords")
        .setOutputCol("words").setStopWords(stop)
    val vectorizer = new CountVectorizer().setInputCol("words").setOutputCol("features")
        .setVocabSize(vocabSize).setMinDF(2)
    val stages = Array( tokenizer, stopremover, vectorizer)
    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(dataset)
    (model, model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary)
  }

  def buildModel(dataset: DataFrame, k: Int, maxIter: Int): LDAModel = {
    val lda = new LDA().
        setK(k).setFeaturesCol("features").setMaxIter(maxIter).setOptimizer("online")
    lda.fit(dataset)
  }

  /// load the corpus of documents and split into train/test
  val dataset = spark.sparkContext.wholeTextFiles(corpusFile).toDF("filename","doc")
  val splits = dataset.randomSplit(Array(0.8, 0.2), 15L)
  val (train, test) = (splits(0), splits(1))

  /// prepare the data and train the LDA model
  val (dataprep, vocab) = buildDataPrepPipeline(train, vocabSize)
  val vectorized = dataprep.transform(train)
  val ldaModel = buildModel(vectorized, k, iter)

  /// print out the topics and perplexity on test data
  val testVect = dataprep.transform(test)
  val perplexity = ldaModel.logPerplexity(ldaModel.transform(testVect))
  println(s"Perplexity=$perplexity")

  /// save topics
  val bc = spark.sparkContext.broadcast(vocab)
  val topicWords = udf { (indices: mutable.WrappedArray[_]) =>
      indices.map { case v: Int => bc.value(v) }
  }
  val topicsOutput = ldaModel.describeTopics().select(topicWords($"termIndices").as("topics"))

  val stringify = udf((vs: Seq[String]) => vs match {
    case null => null
    case _    => s"""[${vs.mkString(",")}]"""
  })

  topicsOutput.withColumn("topics", stringify($"topics")).write.text(outputFile)

}
