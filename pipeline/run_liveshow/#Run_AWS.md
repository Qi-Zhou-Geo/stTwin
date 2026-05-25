## Run the python job as background service

### 0. Make sure you are in the **liveshow** folder
```sh
ls
```

### 1. start the service
```sh
sudo cp t1_main.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable t1_main.service
sudo systemctl start t1_main.service
```

```sh
sudo cp t2_main.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable t2_main.service
sudo systemctl start t2_main.service
```

```sh
sudo cp t3_main.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable t3_main.service
sudo systemctl start t3_main.service
```

### 2. stop the service
```sh
sudo systemctl stop t1_main
```

### 3. check the service
```sh
sudo systemctl status t1_main
```

### 4. check cpu usage
```sh
htop
```
