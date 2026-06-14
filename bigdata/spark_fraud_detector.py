import json
import pandas as pd
import joblib

from pyspark.sql import SparkSession

# Load model
model = joblib.load(
    r"C:\ProgFiles\IntelligentFraudDetection\Outputs\best_rf_model2.pkl"
)

spark = SparkSession.builder \
    .appName("FraudDetector") \
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2"
    ) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "transactions") \
    .load()

raw_df = kafka_df.selectExpr(
    "CAST(value AS STRING) as message"
)

FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13",
    "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


def process_batch(batch_df, batch_id):

    rows = batch_df.collect()

    print("\n" + "=" * 70)
    print(f"BATCH {batch_id}")
    print("=" * 70)

    for row in rows:

        tx = json.loads(row["message"])

        try:

            features = [
                float(tx[col])
                for col in FEATURE_COLUMNS
            ]

            X = pd.DataFrame(
                [features],
                columns=FEATURE_COLUMNS
            )

            fraud_prob = model.predict_proba(X)[0][1]

            prediction = int(fraud_prob >= 0.5)

            risk_score = round(
                fraud_prob * 100,
                2
            )

            print(
                f"Risk Score: {risk_score:6.2f} | "
                f"Fraud Prob: {fraud_prob:.4f} | "
                f"Prediction: {prediction}"
            )

        except Exception as e:

            print("Error processing transaction")
            print("Available keys:", list(tx.keys()))
            print("Error:", e)


query = raw_df.writeStream \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()