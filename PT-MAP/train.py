from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from types import SimpleNamespace
import wrn_mixup_model
from io_utils import parse_args, get_resume_file ,get_assigned_file
from dataset.dataAugment import *
from dataset import HanNomDataset

use_gpu = torch.cuda.is_available()
def train_manifold_mixup(base_loader, base_loader_test, model, start_epoch, stop_epoch):

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    print("stop_epoch", start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        model.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        correct1 = 0.0
        total = 0

        for batch_idx, (input_var, target_var) in enumerate(base_loader):
            
            if use_gpu:
                input_var, target_var = input_var.cuda(), target_var.cuda()
            
            input_var, target_var = Variable(input_var), Variable(target_var)
            lam = np.random.beta(2.0, 2.0)
            _ , outputs , target_a , target_b  = model(input_var, target_var, mixup_hidden= True, mixup_alpha = 2.0 , lam = lam)
            #  mixup_hidden= True
            loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_var.size(0)
            correct += ((lam * predicted.eq(target_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx%32 ==0 :
                print('{0}/{1}'.format(batch_idx,len(base_loader)), 'Loss: %.3f | Acc: %.3f%% '
                             % (train_loss/(batch_idx+1),100.*correct/total))
        
        if not os.path.isdir("weights/ptmap"):
            os.makedirs("weights/ptmap")

        if (epoch % 10==0) or (epoch==stop_epoch-1):
            outfile = os.path.join("weights/ptmap", '{:d}.pt'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict() }, outfile)
        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(base_loader_test):
                if use_gpu:
                    inputs, targets = inputs.to("cuda"), targets.to("cuda")
                inputs, targets = Variable(inputs), Variable(targets)
                f , outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted.eq(targets.data).cpu().sum()).item()

            print('Loss: %.3f | Acc: %.3f%%'
                             % (test_loss/(batch_idx+1), 100.*correct/total ))
        
    return model 

def main(cfg: SimpleNamespace) -> None:
    font_dataset = HanNomDataset(cfg, transform=None, train = True)

    # hyperparameters for training
    start_epoch = cfg.train.start_epoch
    stop_epoch = cfg.train.stop_epoch

    model = wrn_mixup_model.wrn28_10(num_classes=font_dataset.n_chars, loss_type = 'softmax')


    train_dataloader = torch.utils.data.DataLoader(font_dataset,
                                               batch_size=cfg.train.batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               pin_memory=True,
                                               num_workers=2)
    valid_dataloader = torch.utils.data.DataLoader(font_dataset,
                                                batch_size=cfg.train.batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                pin_memory=True,
                                                num_workers=2)

    model = train_manifold_mixup(train_dataloader, valid_dataloader, model, start_epoch, start_epoch+stop_epoch)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str,
                        default='experiment_configs/train_ptmap.yaml',
                        help="Config path")
    parser.add_argument('--epochs', type=int, 
                        default=-1,
                        help='Number of epochs')
    args = parser.parse_args()
    cfg = parse_args(args.cfg_path)
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.train.epochs = cfg.train.epochs if args.epochs <= 0 else args.epochs
    main(cfg)