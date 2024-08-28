# 调用各种在线AI绘图网站的API (简易版MarkDown)
## 兼容A1111webuiAPI的API
https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API

## 环境要求 Python3.10和Redis
## 特性
- 多后端负载均衡
- 不支持并发的后端自动上锁
## 已经适配的后端
- https://github.com/AUTOMATIC1111/stable-diffusion-webui
- https://civitai.com/
- https://fal.ai/models/fal-ai/flux/schnell
- https://replicate.com/black-forest-labs/flux-schnell
- https://www.liblib.art/
- https://tusiart.com/
- https://www.seaart.ai/
- https://www.yunjie.art/
## Q群 575601916
### 部署教程(以下为Windows CMD)
### 需要在服务器上部署Redis服务器！请自行安装
python3.10
```
git clone https://github.com/DiaoDaiaChan/Stable-Diffusion-DrawBridgeAPI
cd Stable-Diffusion-DrawBridgeAPI
```
#### 安装依赖
```
python -m venv venv
.\venv\Scripts\python -m pip install -r .\requirements.txt
```
#### 更改配置文件
复制config_example.yaml为config.yaml
[查看详细说明](DrawBridgeAPI/config_example.yaml)
#### 启动
请注意工作目录必须是DrawBridgeAPI目录!
```
cd DrawBridgeAPI
..\venv\Scripts\python api_server.py --port=8000 --host=127.0.0.1
```
#### 访问
访问 http://localhost:8000/docs# 获取帮助
#### 注意
目前API没有鉴权，请勿将此API暴露在公网，否则可能会被滥用
使用CURL测试
```
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "reimu", "width": 512, "height": 768}' http://localhost:8000/sdapi/v1/txt2img
```
..\venv\Scripts\python api_server.py --port=8000 --host=127.0.0.1

#### 可选服务
启动服务器自带打标服务器
修改config.yaml文件 server_settings - build_in_tagger 为true启动，安装依赖

假如工作路径为 Stable-Diffusion-DrawBridgeAPI\DrawBridgeAPI
```
..\venv\Scripts\python -m pip install -r utils/tagger-requirements.txt
```

### 更新日志
## 2024-08-13
```angular2html

```