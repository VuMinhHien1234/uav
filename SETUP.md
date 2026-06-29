# UAV Pipeline — Hướng dẫn Setup trên GCP Ubuntu VM

## Tổng quan

```
Bước 1   Tạo GCP VM (boot 100GB + 3 disk Ceph OSD)
Bước 2   Cài dependencies (Docker, Python, AWS CLI)
Bước 3   Cài Ceph (cephadm) + 3 OSD + RadosGW
Bước 4   Clone repo + pip install
Bước 5   Khởi động Kafka + MLflow (Docker Compose)
Bước 6   Tạo Ceph user + buckets
Bước 7   (Bỏ qua) Bucket Notification — dùng Kafka direct thay thế
Bước 8   Setup k3s + K8s secrets
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

# Tạo VM (e2-standard-4: 4 vCPU, 16GB RAM, boot 100GB + 3 OSD disks)
gcloud compute instances create uav-pipeline-vm \
  --zone=asia-southeast1-b \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --create-disk=name=ceph-osd-1,size=20GB,type=pd-balanced \
  --create-disk=name=ceph-osd-2,size=20GB,type=pd-balanced \
  --create-disk=name=ceph-osd-3,size=20GB,type=pd-balanced \
  --tags=http-server,https-server

# Mở firewall cho các port cần thiết
gcloud compute firewall-rules create uav-pipeline-ports \
  --allow=tcp:7480,tcp:8443,tcp:5000,tcp:9092 \
  --target-tags=http-server \
  --description="Ceph RadosGW, Dashboard, MLflow, Kafka"

# SSH vào VM
gcloud compute ssh uav-pipeline-vm --zone=asia-southeast1-b
```

> **SSH thủ công (không có gcloud):** Tạo SSH key trên Mac rồi thêm vào VM qua GCP Console → Metadata → SSH Keys.
> ```bash
> ssh-keygen -t ed25519 -f ~/.ssh/gcp_uav -C "YOUR_VM_USERNAME"
> ssh -i ~/.ssh/gcp_uav YOUR_VM_USERNAME@VM_EXTERNAL_IP
> ```

---

## Bước 2 — Cài dependencies

> Chạy toàn bộ trên VM sau khi SSH vào.

```bash
# Update hệ thống
sudo apt-get update && sudo apt-get upgrade -y

# Cài Python 3.11, pip, venv
sudo apt-get install -y python3.11 python3.11-venv python3-pip git curl wget unzip

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

# Kiểm tra cluster
sudo cephadm shell -- ceph -s
```

### 3.2 Thêm 3 OSD (3 disk 20GB vừa tạo)

```bash
# Kiểm tra disk — phải thấy /dev/sdb, /dev/sdc, /dev/sdd
lsblk

# Thêm 3 OSD
sudo cephadm shell -- ceph orch daemon add osd $(hostname):/dev/sdb
sudo cephadm shell -- ceph orch daemon add osd $(hostname):/dev/sdc
sudo cephadm shell -- ceph orch daemon add osd $(hostname):/dev/sdd

# Đợi OSD up
sudo cephadm shell -- ceph osd tree
# Phải thấy 3 OSD đều up

# Kiểm tra cluster healthy
sudo cephadm shell -- ceph -s
# Mong đợi: HEALTH_OK, osd: 3 osds: 3 up
```

### 3.3 Bật RadosGW (S3 endpoint port 7480)

```bash
# Bật RGW trên port 7480
sudo cephadm shell -- ceph orch apply rgw default --port 7480

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
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Test MLflow
curl http://localhost:5000/health
```

---

## Bước 6 — Tạo Ceph user + 4 buckets

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
  ✓ s3://checkpoints
  ✓ s3://training-data
  ✓ s3://mlflow-artifacts
  ✓ s3://fast-weight-state
[Setup] Done!
```

> **Lưu ý:** Không còn bucket `raw-frames` — UAV upload frame thẳng vào `training-data`.

Verify:
```bash
aws --endpoint-url http://localhost:7480 s3 ls
```

---

## Bước 7 — (Đã bỏ) Bucket Notification

> **Không cần chạy bước này.** Pipeline dùng Kafka direct publish — `flight_agent` tự publish event lên Kafka sau mỗi frame upload. Không phụ thuộc vào Ceph bucket notification.

---

## Bước 8 — Setup k3s + K8s secrets

> **Không cần KServe.** Inference chạy trên UAV edge (Jetson Nano). K8s chỉ dùng để spawn training jobs.

### 8.1 Cài k3s (self-managed Kubernetes)

```bash
# Cấu hình local registry trước khi cài k3s
sudo mkdir -p /etc/rancher/k3s
cat << 'EOF' | sudo tee /etc/rancher/k3s/registries.yaml
mirrors:
  "localhost:5001":
    endpoint:
      - "http://localhost:5001"
EOF

# Cài k3s
curl -sfL https://get.k3s.io | sh -

# Nếu thấy "No change detected so skipping service start" → start thủ công
sudo systemctl start k3s

# Cấu hình kubectl không cần sudo
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER ~/.kube/config
sudo chmod 644 /etc/rancher/k3s/k3s.yaml

# Verify
kubectl get nodes
# NAME              STATUS   ROLES                  AGE
# uav-pipeline-vm   Ready    control-plane,master   1m
```

### 8.2 Tạo K8s secret với VM IP thật

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

---

## Bước 9 — Build Docker image cho K8s training job

k3s dùng **containerd** riêng — dùng local registry để bridge với Docker.

```bash
cd ~/uav-pipeline

# Chạy local registry (port 5001, tránh conflict với MLflow 5000)
docker run -d -p 5001:5000 --name local-registry registry:2

# Build image
docker build -t uav-trainer:latest .

# Push vào local registry
docker tag uav-trainer:latest localhost:5001/uav-trainer:latest
docker push localhost:5001/uav-trainer:latest

# k3s pull image từ local registry
sudo k3s ctr images pull --plain-http localhost:5001/uav-trainer:latest

# Verify
sudo k3s ctr images ls | grep uav-trainer
```

---

## Bước 10 — Chạy demo

```bash
cd ~/uav-pipeline
source venv/bin/activate
export PYTHONPATH=$PWD

# Tạo mock data: 5 terrain × 50 frames = 250 frames
python3 scripts/download_mock_data.py
```

Output:
```
Generating mock UAV frames → mock-data/
  5 terrains × 50 frames = 250 total

  ✓ urban      (Urban   )  skip=10  fast=32  medium= 6  slow= 2
  ✓ forest     (Forest  )  skip=20  fast=20  medium= 6  slow= 4
  ✓ desert     (Desert  )  skip=25  fast=15  medium= 8  slow= 2
  ✓ coastal    (Coastal )  skip=20  fast=20  medium= 7  slow= 3
  ✓ mountain   (Mountain)  skip= 0  fast=29  medium=14  slow= 7
```

---

### Demo A — Chạy tự động (toàn bộ pipeline)

Chạy 1 lệnh, demo mặc định terrain `mountain`:

```bash
bash scripts/demo.sh
```

Muốn đổi terrain:
```bash
FLIGHT_TERRAIN=forest bash scripts/demo.sh
```

---

### Demo B — Chạy thủ công từng bước (khuyến nghị khi thuyết trình)

**Terminal 1** — Khởi động consumer (giữ mở suốt):
```bash
cd ~/uav-pipeline
source venv/bin/activate
export PYTHONPATH=$PWD
python3 -m consumers.model_trainer
```

**Terminal 2** — Chạy từng terrain, quan sát kết quả:

```bash
cd ~/uav-pipeline
source venv/bin/activate
export PYTHONPATH=$PWD

# --- Lần 1: Địa hình núi ---
# UAV bay, upload frames, trigger retrain
# → Tạo model: uav-navigator-mountain v1 (bắt đầu từ ImageNet)
FLIGHT_TERRAIN=mountain python3 -m simulator.flight_agent

# --- Lần 2: Địa hình rừng ---
# Terrain khác → model khác hoàn toàn
# → Tạo model: uav-navigator-forest v1
FLIGHT_TERRAIN=forest python3 -m simulator.flight_agent

# --- Lần 3: Bay lại núi ---
# Cùng terrain → fine-tune trên model đã có
# → Tạo model: uav-navigator-mountain v2
#   accuracy cao hơn v1, loss thấp hơn, prev_version=1
FLIGHT_TERRAIN=mountain python3 -m simulator.flight_agent
```

---

### Những gì quan sát được trong demo

**Logs Terminal 2** (flight_agent) — frame-by-frame:
```
Frame 0000 | mountain | 🟢 FAST     | surprise=0.0021  drift=0.002  Wf=0.0012  Wm=0.0004  Ws=0.0001
Frame 0001 | mountain | 🟢 FAST     | surprise=0.0018  drift=0.004  Wf=0.0023  Wm=0.0008  Ws=0.0002
Frame 0005 | mountain | 🟡 MEDIUM   | surprise=0.0042  drift=0.018  Wf=0.0089  Wm=0.0034  Ws=0.0009
Frame 0036 | mountain | 🔴 SLOW     | surprise=0.0058  drift=0.103  Wf=0.1240  Wm=0.0478  Ws=0.0121
```

**Logs Terminal 1** (model_trainer) — training events:
```
Retrain event received: level=slow  key=slow/terrain_mountain/flight_1234_5678.json
Terrain: mountain
K8s Job uav-train-1234 created
K8s Job uav-train-1234 completed
Metrics → accuracy=0.8412  latency=142.3ms  loss=0.2841
PASS — promoting uav-navigator-mountain to Staging
```

**MLflow UI** `http://localhost:5000` — sau khi chạy:

| Run | Terrain | prev_version | accuracy | loss | W_fast_norm |
|-----|---------|-------------|----------|------|-------------|
| uav-retrain-slow-mountain-run1 | mountain | imagenet | ~0.84 | ~0.28 | thấp |
| uav-retrain-slow-forest-run1   | forest   | imagenet | ~0.83 | ~0.30 | thấp |
| uav-retrain-slow-mountain-run2 | mountain | 1        | ~0.89 | ~0.15 | cao hơn |

> Lần 1 (run1): `prev_version=imagenet` → dùng mock metrics (FC chưa được train → pseudo-labels không tin cậy) → promote lên **Staging**.
> Lần 3 (mountain run2): `prev_version=1` → dùng pseudo-labels thật từ flight lần 1 → accuracy/loss thật → nếu pass threshold thì promote lên **Production**.
> `W_fast_norm` / `W_med_norm` / `W_slow_norm` tăng dần — Titans memory tích lũy qua các chuyến bay.

---

### Xem kết quả Ceph sau demo

```bash
# Frames đã upload (theo terrain + flight)
aws --endpoint-url http://localhost:7480 s3 ls s3://training-data/ --recursive

# Checkpoints đã tạo
aws --endpoint-url http://localhost:7480 s3 ls s3://checkpoints/ --recursive

# W matrices (Titans memory state)
aws --endpoint-url http://localhost:7480 s3 ls s3://fast-weight-state/ --recursive
```

### Xem MLflow từ Mac (SSH tunnel)

```bash
# Chạy trên Mac
ssh -i ~/.ssh/gcp_uav -L 5000:localhost:5000 -L 8443:localhost:8443 \
  YOUR_VM_USERNAME@VM_EXTERNAL_IP -N
```

Sau đó mở `http://localhost:5000` trên Mac.

---

## Xem logs

```bash
# model_trainer — training orchestrator
tail -f /tmp/model_trainer.log

# K8s training jobs
kubectl get jobs
kubectl logs job/uav-train-<ID>

# Tất cả K8s pods
kubectl get pods
```

---

## Troubleshooting

**Ceph chưa HEALTH_OK:**
```bash
sudo cephadm shell -- ceph -s
sudo cephadm shell -- ceph health detail
```

**RGW không accessible sau khi start:**
```bash
# Kiểm tra port thật sự đang listen
sudo ss -tlnp | grep 7480

# Kiểm tra container RGW
sudo docker ps | grep rgw
sudo docker logs $(sudo docker ps --format "{{.Names}}" | grep rgw | head -1) 2>&1 | tail -20

# Thử curl
curl http://localhost:7480
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

**`No frames for terrain='xyz'` khi chạy flight_agent:**
```bash
# Chạy lại generate mock data
python3 scripts/download_mock_data.py

# Kiểm tra terrain có trong mock-data/
ls mock-data/
# urban  forest  desert  coastal  mountain
```

**Import error khi chạy Python:**
```bash
# Đảm bảo PYTHONPATH được set
export PYTHONPATH=$PWD   # chạy từ ~/uav-pipeline
```

**k3s không start được / kubectl lỗi permission:**
```bash
sudo systemctl start k3s
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
kubectl get nodes
```

**uav-trainer image không tìm thấy trong k3s:**
```bash
# Push lại vào local registry
docker push localhost:5001/uav-trainer:latest
sudo k3s ctr images pull --plain-http localhost:5001/uav-trainer:latest
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
