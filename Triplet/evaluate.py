from dataset import TripletDataset
from models import init_simsiam_model, init_triplet_model
from dataset.dataAugment import *
from utils import *
import torch
import numpy as np
import faiss

def _topk(sample_labels, test_labels, I, k):
    correct = 0
    for test_label, i in zip(test_labels, I):
        if test_label in i[:min(k, len(i))]:
            correct += 1
    
    acc = correct/float(len(test_labels)+1e-7)
    return acc

def _k_neighbors(cfg, sample_dataset, test_dataset, k, embedding_dim, model_name:str):
    sample_embedding, sample_labels = get_embedding(cfg, model, sample_dataset, model_name, 'sample_dataset')
    test_embedding, test_labels = get_embedding(cfg, model, test_dataset, model_name, 'test_dataset')
    print(test_embedding.shape, test_labels.shape)
    print(sample_embedding.shape, sample_labels.shape)

    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(sample_embedding)
    
    D, I = faiss_index.search(test_embedding, k)
    
    return sample_labels, test_labels, I

def evaluate(cfg, k: int, model, model_name, save_to='visualize.png'):
    model.to(cfg.device)
    sample_dataset = TripletDataset(cfg, transform=test_transforms(cfg), one_font_only=True)
    test_dataset = TripletDataset(cfg, transform=test_transforms(cfg))
    # sanity check mean and std
    sample1 = sample_dataset[0][0]
    sample2 = test_dataset[0][0]
    print('mean and std of the first sample in sample dataset', sample1.mean(), sample1.std())
    print('mean and std of the first sample in test dataset', sample2.mean(), sample2.std())
    
    print("Number of test characters: {}".format(len(test_dataset)))
    print("number of sample characters (used as labels): {}".format(len(sample_dataset)))
    
    knn = _k_neighbors(cfg, sample_dataset, test_dataset, k, model.embedding_dim, model_name)
    acc = _topk(*knn, k)
    print('Top {} accuracy: {:.3f}%'.format(k, acc*100))
    print('Top 1 accuracy: {:.3f}%'.format(_topk(*knn, 1)*100))
    _, _, I = knn
    visualize(I, test_dataset, sample_dataset, model_name+save_to)
    
def visualize(I, test_dataset, sample_dataset, save_to, n=5):
    k = len(I[0])
    imshape = test_dataset[0][0].shape
    imgs = []
    for i in np.random.randint(0, len(test_dataset), size=n):
        img, _ = test_dataset[i]
        imgs.append(img.squeeze().unsqueeze(0).numpy())
        for j in I[i]:
            imgs.append(sample_dataset[j][0].squeeze().unsqueeze(0).numpy())
            
    imgs = np.concatenate(imgs, axis=0)
    imgs = torch.from_numpy(imgs)
    
    from torchvision.utils import save_image
    
    save_image(imgs, save_to, nrow=len(imgs)//n,
            normalize=True, scale_each=True)
        
    
if __name__ == '__main__':
    import argparse
    from utils import parse_args
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, 
                        default='simsiam',
                        help='Name of the training process')
    parser.add_argument('--cfg_path', type=str, 
                        default='experiment_configs/train_simsiam.yaml',
                        help='Config path')
                        
    parser.add_argument('--model_path', type=str, 
                        default='weights/simsiam/pretrained_final.pt',)
                        
    args = parser.parse_args()
    
    cfg = parse_args(args.cfg_path)
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.model.pretrained = False
    
    from models import *
    if 'simsiam' in args.pipeline:
        model = init_simsiam_model(cfg)
    elif 'triplet' in args.pipeline:
        model = init_triplet_model(cfg)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    print('Done loading weights')
    evaluate(cfg, 5, model, args.pipeline)
    