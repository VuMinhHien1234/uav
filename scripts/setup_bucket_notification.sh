#!/bin/bash
# Cấu hình Ceph Bucket Notification → Kafka
# Chạy sau khi Kafka đã up và buckets đã tạo

set -e

KAFKA_BROKER="localhost:9092"
CEPH_ENDPOINT="http://localhost:7480"

echo "[Notification] Enabling pubsub module..."
ceph mgr module enable pubsub

echo "[Notification] Creating SNS topic → Kafka..."
python3 - <<PYEOF
import boto3

sns = boto3.client(
    'sns',
    endpoint_url='${CEPH_ENDPOINT}',
    aws_access_key_id='uavaccess',
    aws_secret_access_key='uavsecret123',
    region_name='us-east-1'
)

# Topic cho raw-frames (UC1/UC3)
resp = sns.create_topic(
    Name='uav-frames',
    Attributes={
        'push-endpoint': 'kafka://${KAFKA_BROKER}',
        'kafka-ack-level': 'broker',
        'use-ssl': 'false',
        'kafka-brokers': '${KAFKA_BROKER}',
        'kafka-topic': 'ai-pipeline'
    }
)
print(f"  raw-frames topic: {resp['TopicArn']}")

# Topic cho checkpoints (UC2)
resp2 = sns.create_topic(
    Name='uav-checkpoints',
    Attributes={
        'push-endpoint': 'kafka://${KAFKA_BROKER}',
        'kafka-ack-level': 'broker',
        'use-ssl': 'false',
        'kafka-brokers': '${KAFKA_BROKER}',
        'kafka-topic': 'ai-pipeline'
    }
)
print(f"  checkpoints topic: {resp2['TopicArn']}")
PYEOF

echo "[Notification] Setting bucket notifications..."
python3 - <<PYEOF
import boto3, json

s3 = boto3.client(
    's3',
    endpoint_url='${CEPH_ENDPOINT}',
    aws_access_key_id='uavaccess',
    aws_secret_access_key='uavsecret123',
    region_name='us-east-1'
)

REGION = "us-east-1"
ACCOUNT = "000000000000"

# raw-frames notification
s3.put_bucket_notification_configuration(
    Bucket='raw-frames',
    NotificationConfiguration={
        'TopicConfigurations': [{
            'TopicArn': f'arn:aws:sns:{REGION}:{ACCOUNT}:uav-frames',
            'Events': ['s3:ObjectCreated:*']
        }]
    }
)
print("  ✓ raw-frames notification set")

# checkpoints notification
s3.put_bucket_notification_configuration(
    Bucket='checkpoints',
    NotificationConfiguration={
        'TopicConfigurations': [{
            'TopicArn': f'arn:aws:sns:{REGION}:{ACCOUNT}:uav-checkpoints',
            'Events': ['s3:ObjectCreated:*']
        }]
    }
)
print("  ✓ checkpoints notification set")
PYEOF

echo ""
echo "[Notification] Done! Verify with:"
echo "  aws --endpoint-url ${CEPH_ENDPOINT} s3api get-bucket-notification-configuration --bucket raw-frames"
