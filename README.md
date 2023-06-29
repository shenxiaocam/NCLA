
Neighbor Contrastive Learning on Learnable Graph Augmentation (NLCA)
====
This repository contains the author's implementation in PyTorch for the paper "Neighbor Contrastive Learning on Learnable Graph Augmentation".

Environment Requirement
===
The experiments were conducted on a single Tesla A40 GPU with 48GB memory. The required packages are as follows:

•	python == 3.6.13

•	torch==1.10.1	

•	numpy==1.19.3

•	scipy==1.5.4

•	scikit_learn==1.1.3

•	dgl==0.9.1


Datasets
===
data/ contains the 5 datasets used in our paper, i.e., Cora, CiteSeer, PubMed, Amazon-Photo, and Coauthor-CS.


Code
===
"gat.py" is the multi-head GAT model.

"train.py" is an example case of the NCLA model for node classification on the Cora dataset.

"loss.py" is the proposed neighbor contrastive loss.

The hyperparameters of NCLA on five datasets can be found in "run.sh".

Plese cite our paper as:
===
Xiao Shen, Dewang sun, Shirui Pan, Xi Zhou, and Laurence T. Yang. Neighbor Contrastive Learning on Learnable Graph Augmentation. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pp. 9782-9791, 2023.

