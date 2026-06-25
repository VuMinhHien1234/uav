#!/bin/bash
# Tạo Ceph credentials + 5 buckets cần thiết
# Chạy 1 lần sau khi Ceph + RadosGW đã up

set -e

echo "[Setup] Creating Ceph S3 user..."
radosgw-admin user create \
  --uid=uavuser \
  --display-name="UAV Pipeline User" \
  --access-key=uavaccess \
  --secret=uavsecret123 2>/dev/null || echo "  (user exists)"

echo "[Setup] Configuring AWS CLI..."
aws configure set aws_access_key_id uavaccess
aws configure set aws_secret_access_key uavsecret123
aws configure set default.region us-east-1
aws configure set default.output json

echo "[Setup] Creating buckets..."
for bucket in raw-frames embeddings checkpoints training-data mlflow-artifacts; do
  aws --endpoint-url http://localhost:7480 s3 mb s3://$bucket 2>/dev/null \
    && echo "  ✓ s3://$bucket" \
    || echo "  (exists) s3://$bucket"
done

echo "[Setup] Verifying..."
aws --endpoint-url http://localhost:7480 s3 ls

echo ""
echo "[Setup] Done!"
