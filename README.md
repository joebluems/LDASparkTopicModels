# LDA and Sentiment Modeling with Spark Models

## Background
We wish to find the topics in a large group of documents using the Spark LDA Implementation. 
The eventual target documents are mostly text (i.e. email threads), so they need to be tokenized & hashed. <br>
To accomplish this, the corpus documents are tokenized (with stop words removed) loaded into vectors. 
After that, an LDA model is trained with a user-specified number of topics (default is 10). Tokens in each topic are shown & 20% of the data is reserved to evaluate the clustering with the perplexity metric.<br>
Furthermore, we wish to build a generic sentiment model on labeled data to apply to unlabeled data using count vectors as features and movie review data. Logistic Regression is using to build the sentiment score.

## prepare the environment
git clone https://github.com/joebluems/LDASparkTopicModels.git<br>
cd LDASparkTopicModels <br>
mkdir documents <br>
python create.py ### this will create 1,000+ files in the documents folder <br>
(optional) check spark & scala version by  ./spark-shell <br> 
unzip movie_review.zip  ### creates labeled training data
vi build.sbt  ## change spark and scala version if needed <br>
sbt package ## if successful, should create a jar in ./target/scala-2.11 <br>

## notes on the files
./build.sbt - sbt file for spark/scala imports <br>
./documents - About 3 lines of text from <b>The Long Goodbye</b>, by Raymond Chandler split among 1,000+ files which represents the corpus and is populated by running create.py <br>
./pipeline_shell - contains corresponding scala code that could be used in the spark shell for testing <br>
./src/main/scala/LDA.scala - here's where the topic model code is located <br>
./src/main/scala/Sentiment.scala - where the sentiment code is located <br>

## run the topic modeling program
<b>Usage: spark-submit --class com.mapr.topics.Main ./target/scala-2.11/topicmodeling_2.11-0.1-SNAPSHOT.jar</b> <br>
If you run this without arguments, we assume "./documents" is the corpus location and you will get 10 topics. <br>
<b> Note #1: if you're running this on a MapR cluster, it will look for the pathname relative to maprfs:// and not be able to find corpus. Use the arguments with full pathnames instead.</b> <br>
<b> Note #2: parameters vocab size = 10,000 and # of iterations = 10 are hard-coded</b> <br>
<b> Note #3: runs in local-mode, which is hard-coded for testing. Switch to cluster or yarn mode if needed. </b> <br>

The output should resemble the following: <br>
+----------------------------------------------------------------------------+ <br>
|topics                                                                      | <br>
+----------------------------------------------------------------------------+<br>
|[bucks, say, got, might, came, never, get, time, less, fifty]               |<br>
|[mouth, brief, seeing, job, cost, behind, anyone, reasons, outside, touched]|<br>
|[said, like, got, get, one, back, didn, time, went, looked]                 |<br>
|[like, want, didn, journal, much, old, asked, either, good, need]           |<br>
|[darling, sick, sleep, nobody, always, made, maybe, dead, candy, bottle]    |<br>
|[crime, cop, side, case, sure, away, went, always, shut, want]              |<br>
|[one, paper, ten, huh, couple, folded, looked, guy, office, small]          |<br>
|[shot, bed, little, hands, put, said, away, wade, hand, light]              |<br>
|[said, know, one, didn, marlowe, lennox, potter, sylvia, came, man]         |<br>
|[road, light, valley, around, office, got, said, idle, entrance, lamp]      |<br>
+----------------------------------------------------------------------------+<br>
<br>
Perplexity=8.096375668125528 <br>

<b>Usage: spark-submit --class com.mapr.topics.Main ./target/scala-2.11/topicmodeling_2.11-0.1-SNAPSHOT.jar  <corpus_file_location> <topics></b> <br>
Running with two arguments allows you to specify the location of the corpus and number of topics. <br>

## run the sentiment analysis
<b>Usage: spark-submit --class com.mapr.sentiment.Main ./target/scala-2.11/topicmodeling_2.11-0.1-SNAPSHOT.jar</b> <br>
If you run this without arguments, we assume "movie_review/pos" contains positive examples and "movie_revies/neg" contains negative examples. Also, "./documents" is the unlabeled corpus location which we wanb to score for sentiment. <br>
<b> Note #1: if you're running this on a MapR cluster, it will look for the pathname relative to maprfs:// and not be able to find corpus. Use the arguments with full pathnames instead.</b> <br>
<b> Note #2: runs in local-mode, which is hard-coded for testing. Switch to cluster or yarn mode if needed. </b> <br>

The output should resemble the following: <br>
Accuracy on test set = 0.79948586 <br>
Area-under-ROC = 0.880782447792757 <br>
+------+-----------------------------------------------------------------+<br>
|score |filename                                                         |<br>
+------+-----------------------------------------------------------------+<br>
|0.9959|file:/Users/joeblue/western/text/project3/documents/K8ZVN1VR.text|<br>
|0.9955|file:/Users/joeblue/western/text/project3/documents/GALF3Z4U.text|<br>
|0.9938|file:/Users/joeblue/western/text/project3/documents/XA5EVGQ8.text|<br>
|0.9923|file:/Users/joeblue/western/text/project3/documents/PRW5YCCB.text|<br>
|0.9913|file:/Users/joeblue/western/text/project3/documents/SQ2WFQQP.text|<br>
|0.9911|file:/Users/joeblue/western/text/project3/documents/R0X2T8L6.text|<br>
|0.9906|file:/Users/joeblue/western/text/project3/documents/DQJUGX3S.text|<br>
|0.9892|file:/Users/joeblue/western/text/project3/documents/34UIUIXZ.text|<br>
|0.9887|file:/Users/joeblue/western/text/project3/documents/A3ZJ2DA5.text|<br>
|0.9873|file:/Users/joeblue/western/text/project3/documents/8RISWJ1V.text|<br>
+------+-----------------------------------------------------------------+<br>
only showing top 10 rows<br>


<br>
