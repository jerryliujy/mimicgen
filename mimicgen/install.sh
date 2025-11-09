sudo apt update
sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update
# sudo apt install python3.9 python3.9-venv python3.9-dev 
# however, autodistill requires python3.10+, so we install python3.10 as well (if hasn't)
sudo apt install python3.8 python3.8-venv python3.8-dev

python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cd ../robomimic
pip install -e .

cd ../robosuite
pip install -e .

cd ../mimicgen

sudo apt-get install libegl-dev
export MUJOCO_GL=egl


