import os
import subprocess

os.chdir('/root')

os.environ['PIP_INDEX_URL'] = 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

subprocess.run(['apt', 'update'])
subprocess.run(['apt', 'install', '-y', 'aria2c', 'sshpass'])
subprocess.run(['pip', 'install', 'jupyterlab', '--break-system-packages'])

while True:
    os.system('python app.py & jupyter-lab --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/ --port=65432 --LabApp.allow_origin=* --LabApp.token= --LabApp.base_url=/diao')