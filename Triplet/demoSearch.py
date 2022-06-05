from dataset import TripletDataset
from dataset.dataAugment import *
from utils import *
import numpy as np
from models import *
import torch
import faiss

class Clustering:
    def __init__(self, cfg, model, model_name):
            
        # Representers for each different character in dataset
        # equivalent to the mean of each cluster
        self.transform = test_transforms(cfg)
        self.model = model
        self.model.to(cfg.device)
        self.representers = TripletDataset(cfg, transform=self.transform, one_font_only=True)
        self.embedding = get_embedding(cfg, self.model, self.representers, model_name, 'sample_dataset')[0]
        assert len(self.embedding.shape) == 2
        self.input_size = cfg.data.input_shape
        
        self.model.eval()
        self.device = cfg.device
        
        self.faiss_index = faiss.IndexFlatL2(model.embedding_dim)
        self.faiss_index.add(self.embedding)
        
    def embed_single_image(self, image):
    
        if isinstance(image, np.ndarray):
            assert len(image.shape) == 3, image.shape
            # gray scale and resize given image
            image = preprocess_img(image, self.input_size)
            image = self.transform(np.array(image))
        if not isinstance(image, torch.Tensor):
            raise ValueError("Unsupported input file, recommend using nd.array as input")
        
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(image).cpu().numpy()
            
    def top_k(self, img_embed, k):
        assert len(img_embed.shape) == 2
        
        D, I = self.faiss_index.search(img_embed, k)
        res = [self.representers.getlabel(i) for i in I[0]]
        return res
 

if __name__ == '__main__':
    import argparse
    from utils import parse_args
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        default='simsiam',
                        help='Name of the training process')
    parser.add_argument('--cfg_path', type=str, 
                        default='experiment_configs/train_simsiam.yaml',
                        help='Config path')
                        
    parser.add_argument('--model_path', type=str, 
                        default='weights/simsiam/pretrained_final.pt',)
                        
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--k', type=int, default=5)
    
    args = parser.parse_args()
    
    cfg = parse_args(args.cfg_path)
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.model.pretrained = False
    
    if 'simsiam' in args.model:
        model = init_simsiam_model(cfg)
    elif 'triplet' in args.model:
        model = init_triplet_model(cfg)
        
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    print('Done loading weights')
    
    import cv2
    import numpy as np
    img = np.array(cv2.imread(args.img_path, 1), dtype=np.float32)
    clustering = Clustering(cfg, model, args.model)
    
    print('Top {} possible characters are: {}'.format(args.k, 
        ','.join(clustering.top_k(clustering.embed_single_image(img), args.k))))