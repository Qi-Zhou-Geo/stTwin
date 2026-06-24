```sh
# Last Update: Last modified: 2026-06-24T12:38:36
# Author: Qi Zhou
```

## Prepare the Docker

### 1. Warmup
Make sure you are at the deploy folder:
```sh
ls
# You should see folders like:
# compose-stTwin.yml
# Dockerfile
```
---

### 1. Run the docker compose
```sh
docker compose -f compose-stTwin.yml up -d
# docker compose -f compose-Flow-Alert.yml up -d

# -f compose-stTwin.yml >> use this file instead of default yml
# up >> start the services defined in compose-stTwin.yml
# -d >> detached mode, run in the background
# --build st-image:dev >> rebuild the Docker image before starting
```
---

### 2. Check the status
```sh
docker compose -f compose-stTwin.yml ps
# docker compose -f compose-Flow-Alert.yml ps

# docker compose -f compose-stTwin.yml logs --tail=100 sedcas
# docker compose -f compose-stTwin.yml logs --tail=100 dashboard
# docker compose -f compose-stTwin.yml logs --tail=100 cloudflared
# docker compose -f compose-Flow-Alert.yml logs --tail=100 flow-alert
```
---

### 3. Stop the whole services
```sh
docker compose -f compose-stTwin.yml down
# docker compose -f compose-Flow-Alert.yml down
```
---

### 4. Restart one service
```sh
docker compose -f compose-stTwin.yml restart sedcas
# docker compose -f compose-Flow-Alert.yml restart sedcas
```
---