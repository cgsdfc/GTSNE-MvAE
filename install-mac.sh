# python --version（显示Python 3.10.9）
# conda create --name py310gnn python==3.10
# conda activate py310gnn（终端显示进入(py310gnn)）

conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
MACOSX_DEPLOYMENT_TARGET=12.1 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch torchvision torchaudio
python -c "import torch; print(torch.__version__)"
#（确认torch版本，显示1.13.1）
MACOSX_DEPLOYMENT_TARGET=12.1 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+${cpu}.html
MACOSX_DEPLOYMENT_TARGET=12.1 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+${cpu}.html
MACOSX_DEPLOYMENT_TARGET=12.1 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-geometric
# conda deactivate（可选，退出当前环境）