if __name__ == '__main__':
	from dataset.ChineseDictionary import get_allCharacters
	import random
	import math
	import argparse
	import numpy as np
	import torch
	from PIL import Image
	from torchvision.utils import save_image
	from dataset.imGen.imgen import FontStorage
	from dataset.dataAugment import basic_transforms, augment_transforms
	from utils import parse_args
	
	allCharacters = get_allCharacters()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_path', type=str, 
						default='./experiment_configs/train_triplet.yaml',
                        help='Configuration file path')
						
	parser.add_argument('--save_to', type=str, 
						default='img/augmented_characters.png',
                        help='Generated images will be saved to this')
	
	parser.add_argument('--size', type=int, 
						default=64,
                        help='Size of generated images')
	
	parser.add_argument('--ic', dest='ic', 
						action='store_true', 
                        help='Generate identical characters or not')
						
	parser.add_argument('--nrow', type=int, 
						default=8, 
                        help='Number of image rows in the final result')

	parser.add_argument('--ncol', type=int, 
						default=8, 
                        help='Number of image columns in the final result')
	args = parser.parse_args()
	cfg = parse_args(args.config_path)
	fonts = FontStorage(args.size)

	num_examples = args.nrow * args.ncol
	
	if args.ic:
		char = allCharacters[random.randint(0, len(allCharacters)-1)]
		chars = [fonts.gen_char_img(char, f_idx % len(fonts)) for f_idx in range(num_examples)]
	else:
		chars = [fonts.gen_char_img(allCharacters[random.randint(0, len(allCharacters)-1)], 
				f_idx % len(fonts)) for f_idx in range(num_examples)]
				
	# save the original images to compare later
	original = torch.from_numpy(np.array(chars)).permute([0,3,1,2])	
	
	# basic transformations, such as resize, gaussian noise and totensor 
	# this is done in individual image level
	transform = basic_transforms(cfg)
	chars = [transform(torch.from_numpy(char)).numpy() for char in chars]

	# some more transformations, this is done in batch level
	transform = augment_transforms(cfg)
	chars = torch.from_numpy(np.array(chars))
	chars = transform(chars)
	
	# log results
	compare_chars = torch.cat((original, chars), 0)
	indices = torch.Tensor(np.arange(num_examples).repeat(2)).long()
	indices[np.arange(num_examples)*2+1] += num_examples
	save_image(compare_chars.index_select(0, indices), args.save_to, ncol=args.ncol,
			normalize=True,  scale_each=True)
	print('After the transformations, the results have:')
	print('Mean: {:.2f}\nStd: {:.2f}'.format(chars.mean(), chars.std()))
	print('Max: {:.2}\nMin: {:.2}'.format(chars.max(), chars.min()))