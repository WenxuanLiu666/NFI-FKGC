# NFI
Relation Learning with Neighborhood Fusion Interaction for Few-Shot Knowledge Graph Completion
# Environment
python 3.9  
GPU 3090  
CUDA 11.4  
pytorch 1.10  
tqdm 4.66  
tensorboardx 2.6.2
# Dataset
We use [NELL-One and Wiki-One](https://drive.google.com/drive/folders/1eaF0CkFeDwC5ikIvERnJHAPvHx1gkeNG) to test our NFI, and these datasets were firstly proposed by xiong. The orginal datasets and pretrain embeddings can be downloaded from [xiong's repo](https://github.com/xiong).
# run
## NELL-One
python main.py --dataset NELL-One --data_path ./nell --few 5 --data_form Pre-Train --prefix nell_5shot --max_neighbor 50 --batch_size 1024

## Wiki-One
python main.py --dataset NELL-One --data_path ./wiki --few 5 --data_form Pre-Train --prefix wiki_5shot --max_neighbor 50 --batch_size 1024

