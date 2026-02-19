conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9
conda activate digress

python -c 'from rdkit import Chem'
conda install -c conda-forge graph-tool=2.45
python -c 'import graph_tool as gt'

conda install -c "nvidia/label/cuda-11.8.0" cuda
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install -e .
g++ -O2 -std=c++11 -o orca orca.cpp
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=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



# python --version
# python -m pip install -U pip --quiet
# python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 --quiet

# TORCH=2.0.1
# CUDA=cpu

# echo "TORCH=$TORCH  CUDA_TAG=$CUDA"

# python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cpu.html --quiet
# python -m pip install torch-sparse   -f https://data.pyg.org/whl/torch-2.0.1+cpu.html --quiet
# python -m pip install torch-geometric --quiet

# python -m pip install -r requirements.txt --quiet
# python -m pip install -e . --quiet