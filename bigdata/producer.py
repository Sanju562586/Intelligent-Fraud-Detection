import pandas as pd
import json
import time
import random
from kafka import KafkaProducer

# Load test data
df = pd.read_csv(r"C:\ProgFiles\IntelligentFraudDetection\test_data.csv")

# Split normal and fraud transactions
normal_df = df[df["Class"] == 0]
fraud_df = df[df["Class"] == 1]

# Kafka producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("Streaming transactions...")

while True:

    # 5% fraud probability
    if random.random() < 0.05:
        row = fraud_df.sample(n=1).iloc[0]
    else:
        row = normal_df.sample(n=1).iloc[0]

    message = row.to_dict()

    producer.send(
        "transactions",
        value=message
    )

    print(
        f"Sent | Class={int(message['Class'])}"
    )

    # 10 transactions per second
    time.sleep(1)