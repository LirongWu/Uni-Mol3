export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /wulirong/Uni-Mol3
python -B preprocess.py --dataset ./data/USPTO_50k --split test --task_type reverse --mode save --data_num 5004

python -B tokenization.py --data-path ./data/USPTO_50k/_data_test/ --weight-path ./model/Tokenizer/checkpoint.pt --save-path 3d_smiles --split train_0 --data-mode 0 --batch-size 4

python -B preprocess.py --dataset ./data/USPTO_50k --split test --task_type reverse --mode load --data_num 5004
python -B preprocess.py --dataset ./data/USPTO_50k --split test --task_type reverse --mode merge --data_num 5004
