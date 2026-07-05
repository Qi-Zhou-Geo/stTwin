```sh
# Last Update: Last modified: 2026-06-19T17:16:48
# Author: Qi Zhou
```

## Quick test mode
```sh
cloudflared tunnel --url http://127.0.0.1:8050 --protocol http2
```

## Proper production mode 
### Note: we do not have domain now
```sh
cloudflared tunnel create stTwin-dash
cloudflared tunnel route dns stTwin-dash dash.yourdomain.com
cloudflared tunnel run stTwin-dash
```