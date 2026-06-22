```sh
# Last Update: Last modified: 2026-06-22T18:57:01
# Author: Qi Zhou
```

## Prepare the Docker

### 1. Warmup
Make sure you are at the deploy folder:
```sh
ls
# You should see folders like:
# docker-compose.yml
# Dockerfile
```
---

### 1. Run the docker compose
```sh
docker compose up -d --build
# up >> start the services defined in docker-compose.yml
# -d >> detached mode, run in the background
# --build st-image:dev >> rebuild the Docker image before starting
```
---

### 2. Check the status
```sh
docker compose ps
# docker compose logs --tail=100 sedcas
# docker compose logs --tail=100 dashboard
# docker compose logs --tail=100 cloudflared
```
---

### 3. Stop the whole services
```sh
docker compose down
```
---

### 4. Restart one service
```sh
docker compose restart sedcas
```
---