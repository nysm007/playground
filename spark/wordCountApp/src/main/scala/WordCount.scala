import org.apache.spark.SparkContext;
import org.apache.spark.SparkContext._;
import org.apache.spark.SparkConf

object WordCount {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("WordCount")
        val sc = new SparkContext(conf)
        val textFile = sc.textFile("/home/bruceyoung/Playground/spark/wordCountApp/sparktest.md")
        val words = textFile.flatMap(line => line.split(" "))
        val wordPairs = words.map(word => (word, 1))
        val wordCounts = wordPairs.reduceByKey((a, b) => a + b)
        println("WordCounts: ")
        wordCounts.collect().foreach(x => println(x))
        wordCounts.saveAsTextFile("/home/bruceyoung/sparkSubmitResult.txt")
    }
}
