# UAV Pipeline — Hướng dẫn Setup trên GCP Ubuntu VM

## Tổng quan

```
Bước 1   Tạo GCP VM
Bước 2   Cài dependencies (Docker, Python, AWS CLI)
Bước 3   Cài Ceph (cephadm) + RadosGW
Bước 4   Clone repo + pip install
Bước 5   Khởi động Kafka + MLflow (Docker Compose)
Bước 6   Tạo Ceph buckets
Bước 7   Cấu hình Bucket Notification → Kafka
Bước 8   Setup k3s + KServe + K8s secrets
Bước 9   Build Docker image cho K8s training job
Bước 10  Chạy demo
```

---

## Bước 1 — Tạo GCP VM

> Cài [gcloud CLI](https://cloud.google.com/sdk/docs/install) trên máy local trước.

```bash
# Đăng nhập
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Tạo VM (e2-standard-4: 4 vCPU, 16GB RAM)
gcloud compute instances create uav-pipeline-vm \
  --zone=asia-southeast1-b \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --create-disk=name=ceph-osd,size=20GB,type=pd-balanced \
  --tags=http-server,https-server

# Mở firewall cho các port cần thiết
gcloud compute firewall-rules create uav-pipeline-ports \
  --allow=tcp:7480,tcp:8443,tcp:5000,tcp:9092 \
  --target-tags=http-server \
  --description="Ceph RadosGW, Dashboard, MLflow, Kafka"

# SSH vào VM
gcloud compute ssh uav-pipeline-vm --zone=asia-southeast1-b
```

---

## Bước 2 — Cài dependencies

> Chạy toàn bộ trên VM sau khi SSH vào.

```bash
# Update hệ thống
sudo apt-get update && sudo apt-get upgrade -y

# Cài Python 3.11, pip, venv
sudo apt-get install -y python3.11 python3.11-venv python3-pip git curl wget

# Cài Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Cài Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version

# Cài AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install
aws --version

# Cài cephadm
curl --silent --remote-name --location \
  https://github.com/ceph/ceph/raw/quincy/src/cephadm/cephadm
chmod +x cephadm
sudo mv cephadm /usr/local/bin/cephadm
cephadm --version
```

---

## Bước 3 — Cài Ceph + RadosGW

### 3.1 Bootstrap Ceph single-node

```bash
# Bootstrap
sudo cephadm bootstrap \
  --mon-ip $(hostname -I | awk '{print $1}') \
  --initial-dashboard-user admin \
  --initial-dashboard-password adminpassword \
  --skip-monitoring-stack

# Thêm shell alias
sudo cephadm shell -- ceph -s   # kiểm tra cluster health
# Hoặc dùng trực tiếp:
alias ceph='sudo cephadm shell -- ceph'
```

### 3.2 Thêm OSD (disk /dev/sdb — disk 20GB vừa tạo)

```bash
# Kiểm tra disk
lsblk

# Thêm OSD
sudo cephadm shell -- ceph orch daemon add osd $(hostname):/dev/sdb

# Kiểm tra OSD up
sudo cephadm shell -- ceph osd tree
```

### 3.3 Fix replication size cho single-node

> Single VM chỉ có 1 OSD. Mặc định Ceph yêu cầu 3 bản sao (size=3) → PGs không active → RGW bị treo.
> Fix: giảm xuống size=1 (không replication, phù hợp cho dev/test).

```bash
# Giảm replication size
sudo cephadm shell -- ceph config set global osd_pool_default_size 1
sudo cephadm shell -- ceph config set global osd_pool_default_min_size 1

# Fix tất cả pools hiện tại (bao gồm .mgr và các pool RGW tạo ra)
sudo cephadm shell -- bash -c "
  for pool in \$(ceph osd pool ls); do
    ceph osd pool set \$pool size 1 --yes-i-really-mean-it 2>/dev/null
    ceph osd pool set \$pool min_size 1 2>/dev/null
  done
  ceph -s
"

# Đợi cluster healthy
watch sudo cephadm shell -- ceph -s
# Ctrl+C khi thấy pgs: active+clean
```

### 3.4 Bật RadosGW (S3 endpoint port 7480)

```bash
# Bật RGW
sudo cephadm shell -- ceph orch apply rgw default

# Đợi 30s rồi kiểm tra
sleep 30
sudo cephadm shell -- ceph orch ps --daemon-type rgw

# Test endpoint
curl http://localhost:7480
# Nếu thấy XML response là OK
```

---

## Bước 4 — Clone repo + cài Python packages

```bash
# Clone repo (thay YOUR_USERNAME/YOUR_REPO)
git clone https://github.com/YOUR_USERNAME/uav-pipeline.git ~/uav-pipeline
cd ~/uav-pipeline

# Tạo virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Cài packages
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python3 -c "import torch, boto3, mlflow, kafka; print('OK')"
```

---

## Bước 5 — Khởi động Kafka + MLflow

```bash
cd ~/uav-pipeline

# Lấy VM IP (dùng cho MLflow → Ceph connection)
VM_IP=$(hostname -I | awk '{print $1}')
echo "VM IP: $VM_IP"

# Cập nhật docker-compose.yml với VM IP thật
# (thay host.docker.internal bằng VM IP để container reach được Ceph)
sed -i "s|http://host.docker.internal:7480|http://${VM_IP}:7480|g" docker-compose.yml

# Khởi động
docker-compose up -d

# Kiểm tra
docker-compose ps
# Cả 3 services (zookeeper, kafka, mlflow) phải ở trạng thái Up

# Test Kafka
docker exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092

# Test MLflow
curl http://localhost:5000/health
```

---

## Bước 6 — Tạo Ceph user + 5 buckets

```bash
cd ~/uav-pipeline
source venv/bin/activate

bash scripts/setup_buckets.sh
```

Output mong đợi:
```
[Setup] Creating Ceph S3 user...
[Setup] Configuring AWS CLI...
[Setup] Creating buckets...
  ✓ s3://raw-frames
  ✓ s3://embeddings
  ✓ s3://checkpoints
  ✓ s3://training-data
  ✓ s3://mlflow-artifacts
[Setup] Done!
```

Verify:
```bash
aws --endpoint-url http://localhost:7480 s3 ls
```

---

## Bước 7 — Cấu hình Bucket Notification → Kafka

```bash
cd ~/uav-pipeline
source venv/bin/activate

bash scripts/setup_bucket_notification.sh
```

Output mong đợi:
```
[Notification] Enabling pubsub module...
[Notification] Creating SNS topic → Kafka...
  raw-frames topic: arn:aws:sns:us-east-1:000000000000:uav-frames
  checkpoints topic: arn:aws:sns:us-east-1:000000000000:uav-checkpoints
[Notification] Setting bucket notifications...
  ✓ raw-frames notification set
  ✓ checkpoints notification set
```

Verify:
```bash
aws --endpoint-url http://localhost:7480 s3api \
  get-bucket-notification-configuration --bucket raw-frames
```

---

## Bước 8 — Setup k3s + KServe + K8s secrets

### 8.1 Cài k3s (self-managed Kubernetes)

k3s là Kubernetes nhẹ, cài 1 lệnh, chạy thẳng trên VM — không cần Docker bên trong như minikube.

```bash
# Cài k3s
curl -sfL https://get.k3s.io | sh -

# Cấu hình kubectl không cần sudo
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER ~/.kube/config

# Verify
kubectl get nodes
# NAME              STATUS   ROLES                  AGE
# uav-pipeline-vm   Ready    control-plane,master   1m
```

### 8.2 Cài KServe

```bash
# Cài cert-manager (KServe cần)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
kubectl wait --for=condition=Ready pod -l app=cert-manager -n cert-manager --timeout=120s

# Cài KServe
kubectl apply -f https://github.com/kserve/kserve/releases/latest/download/kserve.yaml
kubectl wait --for=condition=Ready pod -l control-plane=kserve-controller-manager -n kserve --timeout=120s
```

### 8.3 Tạo K8s secret với VM IP thật

```bash
VM_IP=$(hostname -I | awk '{print $1}')

# Xóa secret cũ nếu có
kubectl delete secret ceph-s3-secret 2>/dev/null || true

# Tạo secret mới với IP thật (pods không dùng localhost được)
kubectl create secret generic ceph-s3-secret \
  --from-literal=AWS_ACCESS_KEY_ID=uavaccess \
  --from-literal=AWS_SECRET_ACCESS_KEY=uavsecret123 \
  --from-literal=MLFLOW_S3_ENDPOINT_URL=http://${VM_IP}:7480 \
  --from-literal=MLFLOW_TRACKING_URI=http://${VM_IP}:5000

# Verify IP đúng
kubectl get secret ceph-s3-secret -o jsonpath='{.data.MLFLOW_TRACKING_URI}' | base64 -d
# Phải ra: http://<VM_IP>:5000 (không phải localhost)
```

### 8.4 Tạo KServe InferenceService placeholder

```bash
cat <<EOF | kubectl apply -f -
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: uav-navigator
  namespace: default
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      storageUri: s3://mlflow-artifacts/placeholder
EOF

kubectl get inferenceservice uav-navigator
```

---

## Bước 9 — Build Docker image cho K8s training job

k3s dùng **containerd** thay vì Docker — cần import image vào containerd.

```bash
cd ~/uav-pipeline

# Build image bằng Docker
docker build -t uav-trainer:latest .

# Export từ Docker rồi import vào containerd của k3s
docker save uav-trainer:latest | sudo k3s ctr images import -

# Verify image có trong containerd
sudo k3s ctr images ls | grep uav-trainer
# Phải thấy: docker.io/library/uav-trainer:latest
```

---

## Bước 10 — Chạy demo

```bash
cd ~/uav-pipeline
source venv/bin/activate

# Set PYTHONPATH để Python tìm được các modules
export PYTHONPATH=$PWD

# Tạo mock data (90 ảnh, 3 môi trường)
python3 scripts/download_mock_data.py

# Chạy demo đầy đủ
bash scripts/demo.sh
```

Output mong đợi:
```
============================================================
 UAV Nested Learning Demo
============================================================

[0] Checking services...
  ✓ Ceph RadosGW OK
  ✓ MLflow OK
  ✓ Kafka OK
  ✓ K8s (k3s) OK
  ✓ uav-trainer image OK

[1] Generating mock frames...
  ✓ env1 (Urban):  30 frames
  ✓ env2 (Forest): 30 frames
  ✓ env3 (Desert): 30 frames

[2] Starting consumers (background)...
  UC1 PID: 12345  (tail /tmp/uc1.log)
  UC2 PID: 12346  (tail /tmp/uc2.log)

[3] Starting UC3 Simulator...
Frame 0000 | env1 | 🟢 FAST         | delta=0.000  drift=0.0
Frame 0001 | env1 | ⚫ SKIP         | delta=0.023  drift=0.0
Frame 0002 | env1 | 🟢 FAST         | delta=0.187  drift=0.2
...
Frame 0030 | env2 | 🟡 MEDIUM       | delta=0.521  drift=4.3
...
Frame 0060 | env3 | 🔴 SLOW         | delta=0.489  drift=50.1

[4] Results...
  --- Ceph Buckets ---
  raw-frames:        45 objects
  embeddings:        45 objects
  checkpoints:        2 objects
  mlflow-artifacts:  12 objects

  --- MLflow ---
  Run: a1b2c3d4...  acc=0.8412  lat=142.3ms
  Run: e5f6g7h8...  acc=0.7891  lat=167.1ms

============================================================
 Demo complete!
 MLflow UI:        http://localhost:5000
 Ceph dashboard:   https://localhost:8443  (admin/adminpassword)
============================================================
```

---

## Xem logs từng consumer

```bash
# UC1 — frame processor
tail -f /tmp/uc1.log

# UC2 — checkpoint validator
tail -f /tmp/uc2.log

# K8s jobs
kubectl get jobs
kubectl logs job/uav-train-<ID>
```

---

## Troubleshooting

**Ceph chưa HEALTH_OK:**
```bash
sudo cephadm shell -- ceph -s
sudo cephadm shell -- ceph health detail
```

**MLflow không connect được Ceph (artifact upload fail):**
```bash
# Kiểm tra VM_IP trong docker-compose.yml đúng chưa
cat docker-compose.yml | grep MLFLOW_S3
# Nếu vẫn là host.docker.internal → chạy lại sed command ở Bước 5
```

**K8s Job fail vì localhost:**
```bash
# Kiểm tra secret
kubectl get secret ceph-s3-secret -o jsonpath='{.data.MLFLOW_TRACKING_URI}' | base64 -d
# Phải là http://<VM_IP>:5000, không phải localhost
```

**Import error khi chạy Python:**
```bash
# Đảm bảo PYTHONPATH được set
export PYTHONPATH=$PWD   # chạy từ ~/uav-pipeline
```

**k3s không start được / kubectl lỗi:**
```bash
# Kiểm tra k3s service
sudo systemctl status k3s

# Xem logs
sudo journalctl -u k3s -n 50

# Restart k3s
sudo systemctl restart k3s

# Kiểm tra kubeconfig đúng chưa
kubectl config view
```

**Image không tìm thấy trong k3s (ErrImageNeverPull):**
```bash
# Import lại image vào containerd
docker save uav-trainer:latest | sudo k3s ctr images import -
sudo k3s ctr images ls | grep uav-trainer
```

---

## Dọn dẹp sau demo

```bash
# Stop containers
docker-compose down

# Stop k3s (nếu cần)
sudo systemctl stop k3s

# Xóa mock data
rm -rf ~/uav-pipeline/mock-data/
```
