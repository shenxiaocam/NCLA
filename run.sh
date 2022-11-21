#!/usr/bin/env sh


echo "=====Cora====="
python train.py --dataset=cora --lr=0.01 --num-heads=4 --num-hidden=32 --tau=1 --weight-decay=1e-4 --in-drop=0.6 --attn-drop=0.5 --epochs=2000 --num-layers=1 --seed=1 --negative-slope=0.2

wait
echo "=====CiteSeer====="
python train.py --dataset=citeseer --lr=0.01 --num-heads=4 --num-hidden=32 --tau=5 --weight-decay=1e-4 --in-drop=0.6 --attn-drop=0.6 --epochs=2000 --num-layers=1 --seed=1 --negative-slope=0.2

wait
echo "=====PubMed====="
python train.py --dataset=pubmed --lr=0.001 --num-heads=2 --num-hidden=32 --tau=5 --weight-decay=5e-05 --in-drop=0. --attn-drop=0.5  --epochs=2000 --num-layers=1 --seed=1 --negative-slope=0.2

wait
echo "=====Coauthor-CS====="
python train.py --dataset=cs --lr=0.05 --num-heads=4 --num-hidden=32 --tau=1 --weight-decay=1e-04 --in-drop=0. --attn-drop=0.5  --epochs=2000 --num-layers=1 --seed=1 --negative-slope=0.2

wait
echo "=====Amazon-Photo====="
python train.py --dataset=photo --lr=0.001 --num-heads=2 --num-hidden=32 --tau=1 --weight-decay=1e-04 --in-drop=0.5 --attn-drop=0.7 --epochs=2000 --num-layers=1 --seed=1 --negative-slope=0.2
