import os
from types import SimpleNamespace

import torch
from torch.utils.tensorboard import SummaryWriter

from models import TripletModel
from dataset.dataAugment import *
from dataset import HanNomDataset
from pytorch_metric_learning import samplers
from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from utils import parse_args, eval_metric_model

def main(cfg: SimpleNamespace) -> None:
    model = TripletModel(   
        backbone=cfg.model.backbone,
        embedding_dim=cfg.model.embedding_dim,
        pretrained=cfg.model.pretrained,
        freeze=cfg.model.freeze
    )

    if cfg.model.weights_path != "":
        model.encoder.load_state_dict(torch.load(cfg.model.weights_path))

    model = model.to(cfg.device)

    opt = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.train.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.train.weight_decay
    )

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=cfg.train.loss_margin, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=cfg.train.loss_margin, distance=distance, type_of_triplets="semihard"
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision",
                                                      'mean_average_precision_at_r', 'r_precision'), k=None)

    
    train_dataset = HanNomDataset(cfg)
    # print("=====Debug Start=========")
    # print(train_dataset.label_list)
    # print(cfg.data.sample_per_cls)
    # print("=====Debug End=========")
    
    query_transform = test_transforms(cfg)
    query_dataset = HanNomDataset(cfg, transform=query_transform, one_font_only=True)
    
    eval_transform = test_transforms(cfg)
    eval_dataset = HanNomDataset(cfg, transform=eval_transform, one_font_only=True)
    
    train_sampler = samplers.MPerClassSampler(train_dataset.label_list, cfg.data.sample_per_cls, batch_size=None,
                                              length_before_new_iter=len(train_dataset.label_list))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.train.batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   pin_memory=True,
                                                   num_workers=torch.multiprocessing.cpu_count())

    data_aug = augment_transforms(cfg=cfg)

    writer = SummaryWriter()

    n_iter = 0
    for epoch in range(cfg.train.epochs):
        
        for batch, (x, y) in enumerate(train_dataloader):
            opt.zero_grad()

            x, y = x.to(cfg.device), y.to(cfg.device)
            x = data_aug(x)
            embedding = model(x)
 
            indices_tuple = mining_func(embedding, y)
            loss = loss_func(embedding, y, indices_tuple)
            loss.backward()
            opt.step()

            if n_iter % cfg.train.log_interval == 0:
                writer.add_scalar(tag="loss", scalar_value=float(loss), global_step=n_iter)
                print(
                    "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                        epoch, n_iter, loss, mining_func.num_triplets
                    )
                )
            if n_iter % cfg.train.eval_inter == 0:
                _ = eval_metric_model(query_dataset, eval_dataset, model, accuracy_calculator, writer, n_iter)
            n_iter += 1    
       
    # save model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(dir_path, 'weights', 'triplet')
    if not (os.path.exists(weight_path)):
        os.makedirs(weight_path)
    torch.save(model.state_dict(), os.path.join(weight_path, cfg.model.name + "_final.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str,
                        default='experiment_configs/train_triplet.yaml',
                        help="Config path")
    parser.add_argument('--epochs', type=int, 
                        default=-1,
                        help='Number of epochs')
    args = parser.parse_args()
    cfg = parse_args(args.cfg_path)
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.train.epochs = cfg.train.epochs if args.epochs <= 0 else args.epochs
    main(cfg)
