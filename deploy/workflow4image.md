```sh
# Last Update: Last modified: 2026-06-22T09:47:49
# Author: Qi Zhou
```

## Prepare the Docker

### 1. Warmup
Make sure you are at the project folder:
```sh
ls
# You should see folders like:
# config
# data
# docs
# ...
```
---

### 2. Build Docker image
Make sure your Docker desktop App is running.

```sh
docker build -f deploy/Dockerfile -t st-image:dev .
# docker build >> create a Docker image
# -f deploy/Dockerfile >> use this specific Dockerfile
# -t st-image:dev >> name the image "st-image", tag it as "dev"
# . >> use current directory as the build context (project root)
```
---

### 3. Check your image or package version
DISK USAGE is around 3 GB, not bad.
```sh
docker images
# you will see:
# IMAGE          ID             DISK USAGE   CONTENT SIZE   EXTRA
# st-image:dev   4a901ceeb85c       2.81GB          685MB    
```
---

### 4. Run a .py file in Docker container
enter the container
```sh
docker run --rm -it --entrypoint bash st-image:dev
```

activate env
```sh
micromamba activate env-qz # "env-qz" is the env name in docker container

# install a package: micromamba install -c conda-forge paramiko
```

run script
```sh
python func/toolkit/env_test.py # exit # exit docker
```

exit docker
```sh
exit
```

---


### 5. Develop stage without rebuild
The "code changes" are mounted into the Docker container,
all edits on QZ's machine are instantly visible inside Docker without rebuilding.

```sh
docker run -it --rm \
  -v $(pwd):/project \
  -w /project \
  st-image:dev \
  func/toolkit/env_test.py

# -it >> run container in interactive terminal mode
# --rm >> Automatically removes the container after it exits
# -v $(pwd):/project >> mount local project folder "$(pwd)" into "/project" inside container
# -w /project >> set working directory to "/project"
# st-image:dev >> image name "st-image", and tag name "dev"
# func/toolkit/env_test.py >> script to run (no rebuild needed)
```
---


### 6-1. Release/Publish stage by Github
Export the current environment, then build the image. <br>
If the build passes, proceed to publish.

```sh
mamba env export -n fd --no-builds > config/environment.yml
```
---


### 6-2. Release/Publish stage by Docker Hub
Align this tag version with the GitHub tag. <br>
You can change the image name and tag as needed.<br>
Name the image "st-image", tag it as "dev". 
```sh
docker build -f deploy/Dockerfile -t st-image:v0dot1 .
```
---


### 7. Publish image to Docker Hub
Push the versioned image for full reproducibility.
```sh
docker push st-image:v0dot1
```
---
