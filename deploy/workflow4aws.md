```sh
# Last Update: Last modified: 2026-06-22T17:38:59
# Author: Qi Zhou
```

## Run stTwin in AWS Lightsail


## 0. Create the instance
```sh
Ubuntu
24.04 LTS
2 GB RAM, 2 vCPUs, 60 GB SSD
```


### 1. Connect by ssh and play it like a normal server
```sh
ssh lightsial
# or use passpord
```


### 2. Install the Docker
Step 1: Remove Old Conflicting Packages
```sh
sudo apt remove -y docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc
```

Step 2: Add Docker Official Repository
```sh
sudo apt update
sudo apt install -y ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings

sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc

sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
```

Step 3: Install Docker + Compose Plugin
```sh
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Install
```sh
docker
docker compose
```

Run Docker Without sudo
```sh
sudo usermod -aG docker ubuntu
```


Step 4: Test Docker
```sh
sudo systemctl enable docker
sudo systemctl start docker
```

### 2.  Docker and compose version
```sh
docker --version
docker compose version
```