sudo apt update
sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update
# sudo apt install python3.9 python3.9-venv python3.9-dev 
# however, autodistill requires python3.10+, so we install python3.10 as well (if hasn't)
sudo apt install python3.10 python3.10-venv python3.10-dev

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -m pip install git+https://github.com/facebookresearch/sam2.git
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# pip install autodistill

# YOU SHOULD COMMENT OUT THE FOLLOWING CODE IN AUTODISTILL
# if not os.path.isdir("~/.cache/autodistill/segment_anything_2/segment-anything-2"):
    #     subprocess.run(
    #         [
    #             "git",
    #             "clone",
    #             "https://github.com/facebookresearch/segment-anything-2.git",
    #         ]
    #     )

    #     os.chdir("segment-anything-2")

    #     subprocess.run(["pip", "install", "-e", "."])