from __future__ import print_function
import argparse
import os
import shutil
import time
import random
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
import os,sys,inspect
from library.train_loop import TrainLoop
from library.optmization import Optmization
from library.optmization import ema_model, WeightEMA
from library.model import Conv_EEG
import math
from library.utils import *
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy

parser = argparse.ArgumentParser(description='PyTorch SSL Training')
# Optimization options
parser.add_argument('--dataset', default='SEED-V', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--method', default='PARSE', type=str, metavar='N',
                    help='method name')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=25,
                        help='Number of labeled data')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--T', default=1, type=float,
                    help='pseudo label temperature')
parser.add_argument('--w-da', default=1.0, type=float,
                    help='data distribution weight')
parser.add_argument('--weak-aug', default=0.2, type=float,
                    help='weak aumentation')
parser.add_argument('--strong-aug', default=0.8, type=float,
                    help='strong aumentation')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--init-weight', default=0, type=float)
parser.add_argument('--end-weight',  default=30, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

def load_config(name):
    with open(os.path.join(sys.path[0], name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config('dataset_params.yaml')

class Model(nn.Module):
    def __init__(self, Conv_EEG):
        super(Model, self).__init__()
        self.model = Conv_EEG(args.dataset, args.method)

    def augmentation(self, input, std):

        input_shape =input.size()
        noise = torch.normal(mean=0.5, std=std, size =input_shape)
        noise = noise.to(device)

        return input + noise

    def forward(self, input, compute_model=True):

        if compute_model==False:
            input_s  = self.augmentation(input, args.strong_aug)
            input_w  = self.augmentation(input, args.weak_aug)

            output = (input_s, input_w)

        else:
            if args.method=='PARSE':
                output_c, output_d = self.model(input)
                output = (output_c, output_d)
            else:
                output = self.model(input)
        return output

def ssl_process(Net, X_train, X_test, Y_train, Y_test, n_labeled_per_class, random_seed):

    data_train,  data_test  = np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1)
    label_train, label_test = Y_train, Y_test

    '''the choice of labeled and unlabeled samples'''

    train_labeled_idxs, train_unlabeled_idxs  = train_split(label_train, n_labeled_per_class, random_seed, config[args.dataset]['Class_No'])

    X_labeled,   Y_labeled   = data_train[train_labeled_idxs],   label_train[train_labeled_idxs]
    X_unlabeled, Y_unlabeled = data_train[train_unlabeled_idxs], label_train[train_unlabeled_idxs]* (-1)

    batch_size = args.batch_size

    unlabeled_ratio = math.ceil(len(Y_unlabeled)/ len(Y_labeled))
    max_iterations  = math.floor(len(Y_unlabeled)/batch_size)
    X_labeled, Y_labeled = np.tile(X_labeled, (unlabeled_ratio,1,1)), np.tile(Y_labeled,(unlabeled_ratio,1))
    print(f"Shape of data_test: {data_test.shape}")
    print(f"Shape of label_test: {label_test.shape}")

    train_dataset_labeled   = load_dataset_to_device(X_labeled,   Y_labeled,   batch_size=batch_size,   class_flag=True,  shuffle_flag=True)
    train_dataset_unlabeled = load_dataset_to_device(X_unlabeled, Y_unlabeled, batch_size=batch_size,   class_flag=False, shuffle_flag=True)
    test_dataset = load_dataset_to_device(data_test, label_test, batch_size=min(batch_size, len(data_test)), class_flag=True, shuffle_flag=False)

    result = train(Net, train_dataset_labeled, train_dataset_unlabeled, test_dataset, max_iterations)

    return result

def train(Net, train_dataset_labeled, train_dataset_unlabeled, test_dataset, max_iterations):

    training_params = {'method': args.method, 'batch_size': args.batch_size, 'alpha': args.alpha,
                        'threshold': args.threshold, 'T': args.T,
                        'w_da': args.w_da, 'lambda_u': args.lambda_u}

    test_metric = np.zeros((args.epochs, 1))
    train_loss_epoch =np.zeros((args.epochs, 1))
    for epoch in range(args.epochs):
        start = time.time()
        train_loss_batch = []
        train_acc_batch = []

        if args.method == 'FixMatch':
            ema_Net = ema_model(Model(Conv_EEG).to(device))
            a_optimizer = optim.Adam(Net.parameters(), args.lr)
            ema_optimizer= WeightEMA(Net, ema_Net, alpha=args.ema_decay, lr=args.lr)

        else:
            pass

        Net.train()

        labeled_train_iter    = iter(train_dataset_labeled)
        unlabeled_train_iter  = iter(train_dataset_unlabeled)

        for batch_idx in range(max_iterations):
            try:
                inputs_x, targets_x = labeled_train_iter.next()
            except:
                labeled_train_iter   = iter(train_dataset_labeled)
                inputs_x, targets_x = next(labeled_train_iter)
            try:
                inputs_u, _ = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter  = iter(train_dataset_unlabeled)
                inputs_u, _ = next(unlabeled_train_iter)


            optimizer = optim.Adam(Net.parameters(), args.lr)
            inputs_x, targets_x, inputs_u = inputs_x.to(device), targets_x.to(device, non_blocking=True), inputs_u.to(device)

            optmization_params = {'lr': args.lr, 'current_epoch': epoch, 'total_epochs': args.epochs, 'current_batch': batch_idx, 'max_iterations': max_iterations,
                                 'init_w': args.init_weight, 'end_w': args.end_weight}
            '''
            Training options for various methods
            '''
            if args.method   == 'MixMatch':
                unsupervised_weight =  Optmization(optmization_params).linear_rampup()
                loss = TrainLoop(training_params).train_step_mix(inputs_x, targets_x, inputs_u, Net, optimizer, unsupervised_weight)

            elif args.method == 'FixMatch':

                loss = TrainLoop(training_params).train_step_fix(inputs_x, targets_x, inputs_u, Net, a_optimizer, ema_optimizer)

            elif args.method == 'AdaMatch':
                unsupervised_weight = Optmization(optmization_params).ada_weight()
                reduced_lr = Optmization(optmization_params).decayed_learning_rate()
                optimizer = optim.Adam(Net.parameters(), reduced_lr)
                loss = TrainLoop(training_params).train_step_ada(inputs_x, targets_x, inputs_u, Net, optimizer, unsupervised_weight)

            elif args.method == 'PARSE':
                unsupervised_weight = Optmization(optmization_params).ada_weight()

                loss = TrainLoop(training_params).train_step_parse(inputs_x, targets_x, inputs_u, Net, optimizer, unsupervised_weight)

            else:
                raise Exception('Methods Name Error')

            train_loss_batch.append(loss)

        Net.eval()

        with torch.no_grad():
            test_loss_batch = []
            test_ytrue_batch = []
            test_ypred_batch = []

            for image_batch, label_batch in test_dataset:
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                loss, y_true, y_pred  = TrainLoop(training_params).eval_step(image_batch,label_batch, Net)
                test_loss_batch.append(loss)

                test_ytrue_batch.append(y_true)
                test_ypred_batch.append(y_pred)


            test_ypred_epoch = np.array(test_ypred_batch).flatten()
            test_ytrue_epoch = np.array(test_ytrue_batch).flatten()

        if args.dataset == 'AMIGOS':
            metric = f1_score(test_ytrue_epoch, test_ypred_epoch, average='macro')
        else:
            metric = accuracy_score(test_ytrue_epoch, test_ypred_epoch)

        test_metric[epoch] = metric
        
    return test_metric

def net_init(model):
    '''load and initialize the model'''
    Net = Model(model).to(device)
    Net.apply(WeightInit)
    Net.apply(WeightClipper)

    return Net
if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    '''set random seeds for torch and numpy libraies'''
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    ''' data and label address loader for each dataset '''
    if args.dataset=='SEED':
        c=0

    elif args.dataset =='SEED-IV':
        train_de    = './library/PARSE/DATA/SEED_IV/new/intra_session/train/de/{}_{}.npy'  # Subject_No, Session_No
        test_de     = './library/PARSE/DATA/SEED_IV/new/intra_session/test/de/{}_{}.npy'  # Subject_No, Session_No
        train_label = './library/PARSE/DATA/SEED_IV/new/intra_session/train/label/{}_{}.npy'
        test_label  = './library/PARSE/DATA/SEED_IV/new/intra_session/test/label/{}_{}.npy'

    elif args.dataset == 'SEED-V':
        c=1
    elif args.dataset == 'AMIGOS':
        c=2
    else:
        raise Exception('Datasets Name Error')
    '''A set of random seeds for later use of choosing labeled and unlabeled data from training set'''
    random_seed_arr = np.array([100, 42])

    for seed in tqdm(range(len(random_seed_arr))):
        random_seed = random_seed_arr[seed]

        n_labeled_per_class = args.n_labeled # number of labeled samples need to be chosen for each emotion class

        '''create result directory'''
        directory = './PARSE/{}_result/ssl_method_{}/run_{}/'.format(args.dataset, args.method, seed+1)

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                pass

        dataset_dict = config[args.dataset]
        '''
        Experiment setup for all four pulbic datasets
        '''
        if args.dataset=='SEED':
            c-0
        elif args.dataset == 'SEED-IV':
            acc_array = np.zeros((dataset_dict['Subject_No'], dataset_dict['Session_No'], args.epochs))

            for subject_num in (range(1, dataset_dict['Subject_No']+1)):
                for session_num in range(1, dataset_dict['Session_No']+1):

                    Net = net_init(Conv_EEG)

                    X_train = np.load(train_de.format(subject_num, session_num))
                    X_test  = np.load(test_de.format(subject_num, session_num))

                    X = np.vstack((X_train, X_test))

                    scaler = MinMaxScaler()
                    X = scaler.fit_transform(X)

                    X_train = X[0: X_train.shape[0]]
                    X_test  = X[X_train.shape[0]:]

                    Y_train = np.load(train_label.format(subject_num, session_num))
                    Y_test  = np.load(test_label.format(subject_num, session_num))

                    acc_array[subject_num-1, session_num-1]  =  np.squeeze(ssl_process(Net, X_train, X_test, Y_train, Y_test, n_labeled_per_class, random_seed))

                    torch.cuda.empty_cache()

            np.save(os.path.join(directory, 'acc_labeled_{}.npy'.format(n_labeled_per_class)), acc_array)
        elif args.dataset == 'SEED-V':
            c=1
        elif args.dataset == 'AMIGOS':
            c=2
        else:
            raise Exception('Datasets Name Error')




#