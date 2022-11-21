# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
import random
import torch
import torch.nn.functional as F
import dgl
from gat import GAT

from utils import load_network_data, get_train_data, random_planetoid_splits
from loss import multihead_contrastive_loss
import warnings
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GAT')
parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--epochs", type=int, default=2000,
                    help="number of training epochs")
parser.add_argument("--dataset", type=str, default="cora",
                    help="which dataset for training")
parser.add_argument("--num-heads", type=int, default=4,
                    help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=32,
                    help="number of hidden units")
parser.add_argument("--tau", type=float, default=1,
                    help="temperature-scales")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed")
parser.add_argument("--in-drop", type=float, default=0.6,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.5,
                    help="attention dropout")
parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")

args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

adj2, features, Y = load_network_data(args.dataset)
features[features > 0] = 1
g = dgl.from_scipy(adj2)

if args.gpu >= 0 and torch.cuda.is_available():
    cuda = True
    g = g.int().to(args.gpu)
else:
    cuda = False

features = torch.FloatTensor(features.todense())
f = open('NCLA_' + args.dataset + '.txt', 'a+')
f.write('\n\n\n{}\n'.format(args))
f.flush()

labels = np.argmax(Y, 1)
adj = torch.tensor(adj2.todense())

all_time = time.time()
num_feats = features.shape[1]
n_classes = Y.shape[1]
n_edges = g.number_of_edges()

# add self loop
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)

# create model
heads = ([args.num_heads] * args.num_layers)
model = GAT(g,
            args.num_layers,
            num_feats,
            args.num_hidden,
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.negative_slope)

if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()

# use optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# initialize graph
dur = []
test_acc = 0

counter = 0
min_train_loss = 100
early_stop_counter = 100
best_t = -1

for epoch in range(args.epochs):
    if epoch >= 0:
        t0 = time.time()
    model.train()
    optimizer.zero_grad()
    heads = model(features)
    loss = multihead_contrastive_loss(heads, adj, tau=args.tau)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        heads = model(features)
        loss_train = multihead_contrastive_loss(heads, adj, tau=args.tau)

    # early stop if loss does not decrease for 100 consecutive epochs
    if loss_train < min_train_loss:
        counter = 0
        min_train_loss = loss_train
        best_t = epoch
        torch.save(model.state_dict(), 'best_NCLA.pkl')
    else:
        counter += 1

    if counter >= early_stop_counter:
        print('early stop')
        break

    if epoch >= 0:
        dur.append(time.time() - t0)

    print("Epoch {:04d} | Time(s) {:.4f} | TrainLoss {:.4f} ".
          format(epoch + 1, np.mean(dur), loss_train.item()))

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_NCLA.pkl'))
model.eval()
with torch.no_grad():
    heads = model(features)
embeds = torch.cat(heads, axis=1)  # concatenate emb learned by all heads
embeds = embeds.detach().cpu()

Accuaracy_test_allK = []
numRandom = 20

for train_num in [1, 2, 3, 4, 20]:

    AccuaracyAll = []
    for random_state in range(numRandom):
        print(
            "\n=============================%d-th random split with training num %d============================="
            % (random_state + 1, train_num))

        if train_num == 20:
            if args.dataset in ['cora', 'citeseer', 'pubmed']:
                # train_num per class: 20, val_num: 500, test: 1000
                val_num = 500
                idx_train, idx_val, idx_test = random_planetoid_splits(n_classes, torch.tensor(labels), train_num,
                                                                       random_state)
            else:
                # Coauthor CS, Amazon Computers, Amazon Photo
                # train_num per class: 20, val_num per class: 30, test: rest
                val_num = 30
                idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

        else:
            val_num = 0  # do not use a validation set when the training labels are extremely limited
            idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

        train_embs = embeds[idx_train, :]
        val_embs = embeds[idx_val, :]
        test_embs = embeds[idx_test, :]

        if train_num == 20:
            # find the best parameter C using validation set
            best_val_score = 0.0
            for param in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]:
                LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, C=param)
                LR.fit(train_embs, labels[idx_train])
                val_score = LR.score(val_embs, labels[idx_val])
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_parameters = {'C': param}

            LR_best = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, **best_parameters)

            LR_best.fit(train_embs, labels[idx_train])
            y_pred_test = LR_best.predict(test_embs)  # pred label
            print("Best accuracy on validation set:{:.4f}".format(best_val_score))
            print("Best parameters:{}".format(best_parameters))

        else:  # not use a validation set when the training labels are extremely limited
            LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
            LR.fit(train_embs, labels[idx_train])
            y_pred_test = LR.predict(test_embs)  # pred label

        test_acc = accuracy_score(labels[idx_test], y_pred_test)
        print("test accuaray:{:.4f}".format(test_acc))
        AccuaracyAll.append(test_acc)

    average_acc = np.mean(AccuaracyAll) * 100
    std_acc = np.std(AccuaracyAll) * 100
    print('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        numRandom, average_acc, std_acc, train_num, val_num))
    f.write('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        numRandom, average_acc, std_acc, train_num, val_num))
    f.flush()

    Accuaracy_test_allK.append(average_acc)

f.close()
