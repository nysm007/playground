## run method


```
~$ sbt package
~$ $SPARK_HOME/bin/spark-submit \
  --class WordCount \
  --master local[2] \
  /path/to/wordCountApp/target/scala-2.10/wordcount_2.10-1.0.jar
```
