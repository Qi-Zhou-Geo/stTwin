```sh
# Last Update: Last modified: 2026-08-07T11:43:27
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
# docker compose -f compose-stTwin.yml logs --tail=100 nginx
```
---

### 3. Stop the whole services
```sh
docker compose -f compose-stTwin.yml down
```
---


### 4. Replace a file 
You can replace a py file and restart the service
```sh
# Make sure you are in the directory containing t3_main.py
docker cp t1_main.py deploy-sedcas-1:/app/func/liveshow/t1_main.py
docker cp t2s1_load_cache.py deploy-dashboard-1:/app/func/liveshow/t2s1_load_cache.py
docker cp t2s3_dash_baord.py deploy-dashboard-1:/app/func/liveshow/t2s3_dash_baord.py
```
---


### 5. Restart one service
```sh
docker compose -f compose-stTwin.yml restart sedcas

# or
docker compose -f compose-stTwin.yml restart dashboard
```
---