python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

python3 -m venv .vllm_venv
source .vllm_venv/bin/activate
pip install --upgrade pip
pip install -r requirements-vllm.txt
deactivate

