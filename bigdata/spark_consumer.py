from pyspark.sql import SparkSession

# Create Spark Session
spark = SparkSession.builder \
    .appName("KafkaDebug") \
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2"
    ) \
    .getOrCreate()

# Reduce Spark logs
spark.sparkContext.setLogLevel("ERROR")

# Read stream from Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "transactions") \
    .load()

# Convert Kafka value to string
raw_df = kafka_df.selectExpr(
    "CAST(value AS STRING) as message"
)

# Print raw messages to console
query = raw_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()