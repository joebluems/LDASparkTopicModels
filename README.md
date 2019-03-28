# LDASparkTopicModels

## Background
We wish to find the topics in a large group of documents using the Spark LDA Implementation. 
The eventual target documents are mostly text (i.e. email threads), so they need to be tokenized & hashed. <br>
To accomplish this, the corpus documents are tokenized (with stop words removed) loaded into vectors. 
After that, an LDA model is trained with a user-specified number of topics (default is 10). Tokens in each topic are shown & 20% of the data is reserved to evaluate the clustering with the perplexity metric.

## prepare the environment
git clone https://github.com/joebluems/LDASparkTopicModels.git<br>
cd LDASparkTopicModels <br>
mkdir documents <br>
python create.py ### this will create 1,000+ files in the documents folder
(optional) check spark & scala version by  ./spark-shell <br> 
vi build.sbt  ## change spark and scala version if needed <br>
sbt package ## if successful, should create a jar in ./target/scala-2.11 <br>

## notes on the files
./build.sbt - sbt file for spark/scala imports <br>
./documents - About 3 lines of text from <b>The Long Goodbye</b>, by Raymond Chandler split among 1,000+ files which represents the corpus and is populated by running create.py <br>
./pipeline_shell - contains corresponding scala code that could be used in the spark shell for testing <br>
./src/main/scala/LDA.scala - here's where the code is located <br>

## run the program
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

