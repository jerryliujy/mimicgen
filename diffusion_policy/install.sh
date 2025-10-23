python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
apt-get install -y python3.9-dev

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt