package com.mapr.sentiment 

import scala.collection.mutable
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Main extends App { 
  Logger.getLogger("org").setLevel(Level.WARN)
  var corpusFile = "./documents"
    
  if (args.length == 1) {
      corpusFile = args(0)
  }

  val spark = SparkSession.builder.master("local").appName("example").getOrCreate()
  import spark.implicits._

  /// load the documents - assume random tag for now
  val dataset = spark.sparkContext.wholeTextFiles(corpusFile).toDF("filename","doc")
  val df1 = dataset.withColumn("target", when(rand() > 0.5, 1).otherwise(0))
  val splits = df1.randomSplit(Array(0.8, 0.2), 15L)
  val (train, test) = (splits(0), splits(1))
  val vocabSize = 1000

  /// build the count vector pipeline ////
  val stop = StopWordsRemover.loadDefaultStopWords("english") ++ Array("tr", "td", "div", "class", "table", "html", "div")
  val tokenizer = new RegexTokenizer().setInputCol("doc").
    setOutputCol("rawwords").setGaps(false).setPattern("[a-zA-Z]{3,}")
  val stopremover = new StopWordsRemover().setInputCol("rawwords").
    setOutputCol("words").setStopWords(stop)
  val vectorizer = new CountVectorizer().setInputCol("words").
    setOutputCol("cv").setVocabSize(vocabSize).setMinDF(2)
  val idf = new IDF().setMinDocFreq(5).
    setInputCol(vectorizer.getOutputCol).setOutputCol("features")
  val indexer = new StringIndexer().
    setInputCol("target").setOutputCol("label")
  val lr = new LogisticRegression().setMaxIter(100).setRegParam(0.01)

  val stages = Array( tokenizer, stopremover, vectorizer, idf,indexer,lr)
  val pipeline = new Pipeline().setStages(stages)

  /// execute the pipeline, calculate accuracy & ROC
  val model = pipeline.fit(train)
  val predictions = model.transform(test)
  val evaluatorAUROC = new BinaryClassificationEvaluator().
     setLabelCol("label").setMetricName("areaUnderROC").
     setRawPredictionCol("probability")
  val accuracy = predictions.filter($"label" === $"prediction").count() / (test.count().toFloat)
  val auc = evaluatorAUROC.evaluate(predictions)
  println(s"Accuracy on test set = $accuracy")
  println(s"Area-under-ROC = $auc")

  //// show top 5 highest sentiment
  val vecToSeq = udf((v: Vector) => v.toArray)
  val scores = predictions.withColumn("score", vecToSeq($"probability").getItem(1) )
  scores.sort(desc("score")).select("score","filename").withColumn("score",format_number($"score",4)).show(5,false)
  
}
