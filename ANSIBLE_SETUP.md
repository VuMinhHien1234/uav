# UAV Nested Learning Pipeline — Setup bằng Ansible (3 VM, hyper-converged)

Từ bản 1-VM cũ, project chuyển sang setup **3 VM giống hệt nhau** — mỗi VM chạy đủ
Ceph MON+OSD, Kafka broker+ZooKeeper, k3s server node, RGW+MLflow — để Ceph, Kafka,
k3s đều đạt quorum 3 (chịu được mất đúng 1 VM). Xem giải thích đầy đủ trong lịch sử
trò chuyện/README nếu cần lý do vì sao không tách riêng "1 VM = 1 hệ".

**Thời gian**: ~25–40 phút (bootstrap Ceph + k3s + Kafka/ZK trên cả 3 máy).

> ⚠️ **Chưa được test trên GCP thật.** Playbook này viết từ quy trình multi-node
> chuẩn của cephadm/k3s/Kafka+ZooKeeper nhưng chưa chạy thử trên cluster thật.
> Chạy thử trên 3 VM demo/rẻ trước, đọc kỹ output `ceph -s` / `kubectl get nodes`
> / `kafka-topics` ở mỗi bước trước khi tin tưởng đưa vào dùng thật.

---

## Yêu cầu trước khi bắt đầu

### Trên cả 3 GCP VM
- Ubuntu 22.04
- **1 disk trống** mỗi VM để làm Ceph OSD: `/dev/sdb` (khác bản cũ — trước đây 1 VM
  cần 3 disk, giờ mỗi VM chỉ cần 1 disk vì 3 VM tự cho đủ 3 bản sao)
- Cùng 1 VPC/subnet (để 3 VM thấy IP nội bộ của nhau)
- Đã tạo SSH key và thêm vào cả 3 VM

### Trên Mac
```bash
pip3 install ansible
ansible-galaxy collection install community.docker ansible.posix
ansible --version   # ansible [core 2.14.x] trở lên
```

---

## Bước 1 — Tạo 3 VM trên GCP

> IP ngoài mặc định của GCP là **ephemeral** — đổi mỗi lần bạn stop/start VM
> (khác IP nội bộ, cái này giữ nguyên suốt vòng đời VM). Với cụm 3 VM, mỗi lần
> IP ngoài đổi bạn phải sửa lại `ansible_host` trong `inventory.ini` cho cả 3
> host — nên đặt static IP ngay từ đầu để khỏi phải làm lại việc này mỗi lần
> tắt/bật máy.

```bash
# Đặt trước 3 static external IP (không đổi khi stop/start VM)
for i in 1 2 3; do
  gcloud compute addresses create uav-vm-$i-ip --region=asia-southeast1
done

for i in 1 2 3; do
  gcloud compute instances create uav-vm-$i \
    --zone=asia-southeast1-b \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --create-disk=name=ceph-osd-$i,size=20GB,type=pd-balanced \
    --address=$(gcloud compute addresses describe uav-vm-$i-ip --region=asia-southeast1 --format='get(address)') \
    --tags=http-server,https-server
done

# Lấy lại 3 IP tĩnh này để điền vào ansible_host trong inventory.ini (Bước 2)
for i in 1 2 3; do
  echo "uav-vm-$i: $(gcloud compute addresses describe uav-vm-$i-ip --region=asia-southeast1 --format='get(address)')"
done
```

# Mở firewall NỘI BỘ giữa 3 VM (Ceph MON/OSD, Kafka+ZK, k3s) — chỉ cho subnet,
# không mở ra internet. Thay <subnet-cidr> bằng CIDR subnet thật của bạn.
# Lưu ý: 29092 là port inter-broker Kafka thật sự dùng để 3 broker nói chuyện
# với nhau (không phải 9093 — dự án này dùng ZooKeeper, không phải KRaft, nên
# không có "controller port" riêng).
gcloud compute firewall-rules create uav-cluster-internal \
  --allow=tcp:6789,tcp:3300,tcp:6800-7300,tcp:7480,tcp:9092,tcp:29092,tcp:2181,tcp:2888,tcp:3888,tcp:6443,tcp:8472,tcp:10250,tcp:5000,tcp:5432 \
  --source-ranges=<subnet-cidr>

# Mở port cần truy cập từ Mac (SSH tunnel dùng sau) — theo IP của bạn, không mở đại trà
gcloud compute firewall-rules create uav-cluster-external \
  --allow=tcp:22 \
  --source-ranges=<IP-của-bạn>/32
```

---

## Bước 2 — Cấu hình inventory (3 host)

Mở `ansible/inventory.ini`, điền IP ngoài + username thật cho cả 3 VM:

```ini
[cluster_nodes]
uav-vm-1 ansible_host=<IP_VM1> ansible_user=<username> ansible_ssh_private_key_file=~/.ssh/gcp_uav
uav-vm-2 ansible_host=<IP_VM2> ansible_user=<username> ansible_ssh_private_key_file=~/.ssh/gcp_uav
uav-vm-3 ansible_host=<IP_VM3> ansible_user=<username> ansible_ssh_private_key_file=~/.ssh/gcp_uav

[cluster_nodes:vars]
ansible_python_interpreter=/usr/bin/python3
```

> **VM đầu tiên trong danh sách là VM bootstrap** — nó bootstrap Ceph và
> `k3s --cluster-init`, 2 VM còn lại join vào. Thứ tự trong file này quyết định
> việc đó, không phải bản thân VM nào đặc biệt.

---

## Bước 3 — Cấu hình playbook (nếu cần)

Mở `ansible/playbook.yml`, xem `vars` đầu file:

```yaml
vars:
  git_repo: "https://github.com/YOUR_USERNAME/uav-pipeline.git"  # ← sửa repo của bạn
  s3_access_key: "uavaccess"
  s3_secret_key: "uavsecret123"
  dashboard_password: "adminpassword"
  postgres_password: "mlflowpassword"
  ceph_osd_disk: /dev/sdb          # ← sửa nếu tên disk khác (ví dụ /dev/vdb)
  cloud_sql_private_ip: ""         # để trống = Postgres tự host trên VM bootstrap
                                    # điền IP nội bộ Cloud SQL = dùng managed Postgres (khuyến nghị)
  load_balancer_ip: ""             # điền sau khi tạo GCP internal LB ở Bước 7
```

Kiểm tra tên disk trên từng VM: `ssh <user>@<VM_IP> lsblk`

---

## Bước 4 — Kiểm tra kết nối tới cả 3 VM

```bash
cd ansible/
ansible -i inventory.ini cluster_nodes -m ping
```

Kết quả mong muốn — cả 3 đều `SUCCESS`:
```
uav-vm-1 | SUCCESS => { "ping": "pong" }
uav-vm-2 | SUCCESS => { "ping": "pong" }
uav-vm-3 | SUCCESS => { "ping": "pong" }
```

---

## Bước 5 — Chạy playbook

```bash
cd ansible/
ansible-playbook -i inventory.ini playbook.yml
```

Các giai đoạn (chạy trên cả 3 host, trừ khi ghi rõ "chỉ VM bootstrap"):

```
[1] Cài Docker, Python, AWS CLI, cephadm                    (cả 3 VM)
[2] Ceph: bootstrap trên VM bootstrap, add 2 VM còn lại,
    grow MON lên 3, CRUSH failure domain = host, OSD mỗi VM  (~5-8 phút chờ)
[3] k3s: cluster-init trên VM bootstrap, 2 VM còn lại join
    (HA embedded-etcd)
[4] Clone repo + pip install                                 (cả 3 VM)
[5] Kafka + ZooKeeper 3-node ensemble + MLflow (docker-compose)
[6] Tạo Ceph S3 user + 4 bucket                              (chỉ VM bootstrap)
[7] Tạo K8s secret (Ceph + MLflow)                           (chỉ VM bootstrap)
[8] Build Docker image trên VM bootstrap, mirror sang 2 VM còn lại
[9] Generate mock UAV data                                    (cả 3 VM)
```

Ansible idempotent — chạy lại an toàn nếu lỗi giữa chừng:
```bash
ansible-playbook -i inventory.ini playbook.yml
```

Chỉ chạy từ 1 bước cụ thể — dùng tên task **không chứa** biến Jinja (những task có
`{{ bootstrap_host }}` trong tên sẽ hiển thị tên đã render lúc chạy thật, ví dụ
"...cluster-init on uav-vm-1", không phải chuỗi literal có `{{ }}`, nên hãy chọn 1
task có tên cố định như dưới đây):
```bash
ansible-playbook -i inventory.ini playbook.yml --start-at-task "[3] Fix kubeconfig permissions on every host"
```

---

## Bước 6 — Kiểm tra sau khi setup

SSH vào **VM bootstrap** (VM đầu tiên trong inventory):

```bash
ssh -i ~/.ssh/gcp_uav <user>@<VM1_IP>

cephadm shell -- ceph -s
# Mong đợi: HEALTH_OK, mon: 3 daemons, osd: 3 osds: 3 up

kubectl get nodes
# Mong đợi: cả 3 node đều Ready

docker exec kafka kafka-topics --bootstrap-server localhost:9092 --describe --topic uav-retrain
# Mong đợi: 3 partition

curl http://localhost:7480          # Ceph RGW trên VM này
curl http://localhost:5000/health   # MLflow trên VM này
```

Nếu bất kỳ cái nào không đủ 3 (mon/osd/node/partition) — dừng lại và debug trước
khi chạy demo, đừng bỏ qua.

---

## Bước 7 (tuỳ chọn nhưng khuyến nghị) — Load balancer cho RGW + MLflow

Nếu bỏ qua bước này, mọi client (`config/settings.py`) sẽ trỏ thẳng vào IP của VM
bootstrap — quay lại đúng vấn đề "single VM IP" mà setup 3-VM này nhằm tránh.

```bash
ansible-playbook -i inventory.ini playbook.yml --tags loadbalancer
```

Play này chỉ in ra các lệnh `gcloud` cần chạy (tạo instance group + health check +
backend service + forwarding rule cho port 7480 và 5000) — chạy tay vì phụ thuộc
zone/network cụ thể của bạn. Sau khi có IP của forwarding rule, điền vào
`load_balancer_ip` trong `playbook.yml` rồi chạy lại Bước 5 để mọi endpoint (S3,
MLflow, K8s secret) trỏ qua LB thay vì 1 VM.

---

## Bước 8 — Mở SSH tunnel xem UI từ Mac

```bash
ssh -i ~/.ssh/gcp_uav \
  -L 5000:localhost:5000 \
  -L 8080:localhost:8080 \
  -L 8334:localhost:8334 \
  -L 8443:localhost:8443 \
  -N <user>@<VM1_external_ip>
```

| Service | URL | Tài khoản |
|---|---|---|
| MLflow | http://localhost:5000 | không cần |
| Kafka UI | http://localhost:8080 | không cần |
| Filestash | http://localhost:8334 | S3: uavaccess / uavsecret123 |
| Ceph Dashboard | https://localhost:8443 | admin / adminpassword |

---

## Bước 9 — Chạy demo + consumer/watcher

Consumer và watcher giờ là 2 tiến trình tách rời (xem `consumers/model_trainer.py`,
`consumers/model_trainer_watcher.py`) — cần chạy cả 2:

```bash
ssh -i ~/.ssh/gcp_uav <user>@<VM1_IP>
cd ~/uav && source venv/bin/activate

python3 -m consumers.model_trainer &
python3 -m consumers.model_trainer_watcher &

# Demo bay
bash scripts/demo.sh

# Demo với threshold thấp hơn để thấy FAST/MEDIUM/SLOW rõ hơn
NL_FAST_DELTA=0.0004 NL_MEDIUM_DELTA=0.0007 NL_SLOW_ACCUMULATOR=0.015 \
FLIGHT_TERRAIN=forest bash scripts/demo.sh
```

Muốn tăng song song: chạy thêm `python3 -m consumers.model_trainer` (cùng
`group_id`, Kafka tự chia 3 partition cho các instance) trên VM khác.

### Rollback model nếu bản mới bay tệ hơn

```bash
python3 -m consumers.rollback_model --terrain forest --list       # xem các version
python3 -m consumers.rollback_model --terrain forest               # về version trước đó
python3 -m consumers.rollback_model --terrain forest --version 4   # về đúng version 4
```

---

## Xử lý lỗi thường gặp (multi-node)

### `Permission denied (publickey)`
```bash
ssh-add ~/.ssh/gcp_uav
ssh -i ~/.ssh/gcp_uav -v <user>@<VM_IP>
```

### Ceph: `ceph orch host add` báo lỗi / VM không join được
Thường do SSH key của cephadm chưa được authorize đúng trên VM đó, hoặc firewall
nội bộ chưa mở port 6789/3300. Kiểm tra:
```bash
cephadm shell -- ceph orch host ls     # phải thấy đủ 3 host
```

### k3s: VM join không thành công / `kubectl get nodes` thiếu node
```bash
sudo systemctl status k3s
sudo journalctl -u k3s -n 50
```
Kiểm tra firewall nội bộ đã mở 6443 (API) và 8472 (flannel vxlan) giữa 3 VM chưa.

### Kafka: broker không thấy nhau / `kafka-topics --describe` báo lỗi
Kiểm tra file `.env` trên từng VM (`cd ~/uav && cat .env`) — `KAFKA_ZOOKEEPER_CONNECT`
phải liệt kê đủ IP nội bộ của cả 3 VM, không phải `zookeeper:2181` mặc định.

### `ceph -s` chỉ thấy 1 mon / 1 osd
Playbook chưa chạy xong các task `[2]` trên VM bootstrap, hoặc bị `ignore_errors`
nuốt mất lỗi thật — chạy lại thủ công từng lệnh `ceph orch ...` trong task đó để
xem lỗi gốc.

---

## So sánh bản 1-VM (cũ) vs 3-VM (hiện tại)

| | 1 VM (cũ) | 3 VM (hiện tại) |
|---|---|---|
| Ceph MON | 1 (single point of failure) | 3 (quorum, chịu mất 1) |
| Kafka broker | 1 | 3 + ZK ensemble 3 |
| k3s control plane | 1 node | 3 node HA (embedded etcd) |
| Disk OSD cần | 3 disk trên 1 máy | 1 disk mỗi máy × 3 máy |
| Chịu lỗi | Mất VM = mất tất cả | Mất 1/3 VM vẫn sống |
| Thời gian setup | ~15-25 phút | ~25-40 phút |
