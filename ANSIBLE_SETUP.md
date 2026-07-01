# UAV Nested Learning Pipeline — Setup bằng Ansible

Thay vì chạy từng lệnh thủ công như trong `SETUP.md`, dùng Ansible để tự động hóa toàn bộ quá trình.  
**Thời gian**: ~15–25 phút (chủ yếu chờ Ceph khởi động).

---

## Ansible là gì?

Ansible là công cụ tự động hóa: bạn viết các bước cài đặt vào file YAML (playbook), Ansible SSH vào VM và chạy tất cả theo thứ tự — không cần ngồi gõ từng lệnh.

```
Mac (chạy Ansible) ──SSH──▶ GCP VM (được cài đặt tự động)
```

---

## Yêu cầu trước khi bắt đầu

### Trên GCP VM
- Ubuntu 22.04
- 3 disk trống để làm Ceph OSD: `/dev/sdb`, `/dev/sdc`, `/dev/sdd`
- Đã tạo SSH key và thêm vào VM (xem `SETUP.md` phần SSH)

### Trên Mac
- Python 3 đã cài
- Ansible và collection `community.docker`

---

## Bước 1 — Cài Ansible trên Mac

```bash
pip3 install ansible

# Cài collection để quản lý Docker container
ansible-galaxy collection install community.docker
```

Kiểm tra:
```bash
ansible --version
# ansible [core 2.14.x]
```
---

## Bước 2 — Cấu hình inventory

Mở file `ansible/inventory.ini` và thay 2 giá trị:

```ini
[uav_vm]
uav-pipeline-vm ansible_host=35.185.186.81   # ← IP ngoài của VM trên GCP
                ansible_user=minhvu           # ← username trên VM
                ansible_ssh_private_key_file=~/.ssh/gcp_uav

[uav_vm:vars]
ansible_python_interpreter=/usr/bin/python3
```

> **Tìm IP VM**: GCP Console → Compute Engine → VM instances → cột "External IP"

---

## Bước 3 — Cấu hình playbook (nếu cần)

Mở `ansible/playbook.yml`, xem phần `vars` đầu file:

```yaml
vars:
  git_repo: "https://github.com/YOUR_USERNAME/uav-pipeline.git"  # ← sửa repo của bạn
  s3_access_key: "uavaccess"       # giữ nguyên nếu dùng credentials mặc định
  s3_secret_key: "uavsecret123"
  dashboard_password: "adminpassword"
  ceph_osd_disks:
    - /dev/sdb
    - /dev/sdc
    - /dev/sdd
```

Nếu VM có tên disk khác (ví dụ `/dev/vdb`), sửa lại `ceph_osd_disks` cho đúng.  
Kiểm tra tên disk bằng: `ssh minhvu@VM_IP lsblk`

---

## Bước 4 — Kiểm tra kết nối trước khi chạy

```bash
cd ansible/
ansible -i inventory.ini uav_vm -m ping
```

Kết quả mong muốn:
```
uav-pipeline-vm | SUCCESS => {
    "ping": "pong"
}
```

Nếu lỗi `Permission denied`: kiểm tra lại `ansible_ssh_private_key_file` và `ansible_user`.

---

## Bước 5 — Chạy playbook

```bash
cd ansible/
ansible-playbook -i inventory.ini playbook.yml
```

Quá trình chạy ~15–25 phút, gồm 8 giai đoạn:

```
[1/8] Cài Docker, Python, AWS CLI, cephadm
[2/8] Bootstrap Ceph + OSD + RadosGW      (~3–5 phút chờ)
[3/8] Clone repo + pip install
[4/8] Khởi động Kafka, MLflow, Kafka UI, Filestash
[5/8] Tạo Ceph S3 user + 4 buckets
[6/8] Cài k3s + fix kubeconfig + tạo K8s secret
[7/8] Build Docker image + push lên local registry
[8/8] Generate mock UAV data
```

Ansible in kết quả từng task — `ok` (thành công), `changed` (đã thay đổi), `failed` (lỗi).

### Chạy lại nếu bị lỗi giữa chừng

Ansible **idempotent** — có thể chạy lại mà không bị lỗi:
```bash
ansible-playbook -i inventory.ini playbook.yml
```
Những task đã hoàn thành sẽ báo `ok` và bỏ qua, chỉ chạy lại task bị lỗi.

### Chỉ chạy từ bước nhất định (nếu bị lỗi ở giữa)

```bash
# Chỉ chạy từ task có tên chứa "[6/8]" trở đi
ansible-playbook -i inventory.ini playbook.yml --start-at-task "[6/8] Install k3s"
```

---

## Bước 6 — Kiểm tra sau khi setup xong

Ansible sẽ in thông báo cuối:

```
TASK [Setup complete!]
ok: [uav-pipeline-vm] => {
    "msg": "
    ✓ Ceph RadosGW   → http://10.x.x.x:7480
    ✓ Ceph Dashboard → https://10.x.x.x:8443  (admin / adminpassword)
    ✓ MLflow UI      → http://10.x.x.x:5000
    ✓ Kafka UI       → http://10.x.x.x:8080
    ✓ Filestash      → http://10.x.x.x:8334
    ...
```

Kiểm tra nhanh trên VM:
```bash
ssh -i ~/.ssh/gcp_uav minhvu@VM_IP

# Kiểm tra các service
kubectl get nodes          # phải thấy "Ready"
docker ps                  # phải thấy kafka, mlflow, kafka-ui, filestash
curl http://localhost:7480 # Ceph RadosGW
curl http://localhost:5000/health  # MLflow
```

---

## Bước 7 — Mở SSH tunnel để xem UI từ Mac

> **Chạy lệnh này trên Mac** (terminal mới, không phải bên trong SSH):

```bash
ssh -i ~/.ssh/gcp_uav \
  -L 5000:localhost:5000 \
  -L 8080:localhost:8080 \
  -L 8334:localhost:8334 \
  -L 8443:localhost:8443 \
  -N minhvu@VM_EXTERNAL_IP
```

Sau đó mở trên Mac:

| Service | URL | Tài khoản |
|---|---|---|
| MLflow | http://localhost:5000 | không cần |
| Kafka UI | http://localhost:8080 | không cần |
| Filestash | http://localhost:8334 | S3: uavaccess / uavsecret123 |
| Ceph Dashboard | https://localhost:8443 | admin / adminpassword |

---

## Bước 8 — Chạy demo

```bash
# SSH vào VM
ssh -i ~/.ssh/gcp_uav minhvu@VM_EXTERNAL_IP

cd ~/uav
source venv/bin/activate

# Demo thông thường
bash scripts/demo.sh

# Demo với mock data (thresholds thấp hơn để thấy FAST/MEDIUM/SLOW)
NL_FAST_DELTA=0.0004 \
NL_MEDIUM_DELTA=0.0007 \
NL_SLOW_ACCUMULATOR=0.015 \
FLIGHT_TERRAIN=forest \
bash scripts/demo.sh
```

---

## Sau khi reboot VM

Ansible đã tự động cài service `fix-k3s-permissions` vào systemd.  
Mỗi lần VM khởi động lại, service này tự sửa quyền kubeconfig — **không cần làm thủ công nữa**.

Chỉ cần chạy lại:
```bash
cd ~/uav && docker-compose up -d
```

Kiểm tra:
```bash
kubectl get nodes   # Ready
curl http://localhost:7480   # Ceph OK
curl http://localhost:5000/health  # MLflow OK
```

---

## Xử lý lỗi thường gặp

### Lỗi: `Permission denied (publickey)`
```bash
# Kiểm tra SSH key
ssh-add -l
ssh-add ~/.ssh/gcp_uav

# Thử kết nối trực tiếp
ssh -i ~/.ssh/gcp_uav -v minhvu@VM_IP
```

### Lỗi: Task `[2/8] Bootstrap Ceph` bị skip hoặc fail
Ceph đã được bootstrap từ lần chạy trước (thư mục `/var/lib/ceph` đã tồn tại).  
Đây là bình thường — Ansible bỏ qua vì đã `creates: /var/lib/ceph`.

Nếu muốn bootstrap lại từ đầu (xóa Ceph hoàn toàn):
```bash
# SSH vào VM rồi chạy
sudo cephadm rm-cluster --fsid $(sudo cephadm ls | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['fsid'])") --force
sudo rm -rf /var/lib/ceph
```
Sau đó chạy lại playbook.

### Lỗi: `[6/8] Wait for k3s node to be Ready` timeout
```bash
# SSH vào VM
sudo systemctl status k3s
sudo journalctl -u k3s -n 50
```

### Lỗi: Docker image build fail
Kiểm tra `Dockerfile` và đảm bảo `git_repo` trong `playbook.yml` đúng.

---

## So sánh Ansible vs cài thủ công

| | Thủ công (`SETUP.md`) | Ansible (`ANSIBLE_SETUP.md`) |
|---|---|---|
| Thời gian | ~60–90 phút | ~15–25 phút |
| Cần gõ lệnh | ~50+ lệnh | 1 lệnh |
| Lỗi do gõ sai | Cao | Không có |
| Chạy lại khi lỗi | Phải nhớ đang ở bước nào | Tự tiếp tục từ chỗ lỗi |
| Tái sử dụng cho VM mới | Phải làm lại từ đầu | Chạy lại playbook là xong |
