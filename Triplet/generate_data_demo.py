if __name__ == '__main__':
    from dataset.ChineseDictionary import get_allCharacters
    import random
    import math
    import argparse
    import numpy as np
    import torch
    from torchvision.utils import save_image
    from dataset.imGen.imgen import FontStorage
    
    allCharacters = get_allCharacters()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ic', dest='ic', 
                        action='store_true', 
                        help='Generate identical characters or not')
    parser.add_argument('--save_to', type=str, 
                        default='img/generated_characters.png',
                        help='Generated images will be saved to this')
    parser.add_argument('--size', type=int, 
                        default=64,
                        help='Size of generated images')
    parser.add_argument('--font_name', dest='font_name', 
                        action='store_true',
                        help='Whether to display font names or not')
    args = parser.parse_args()
    
    num_examples = len(fonts)
    fonts = FontStorage(img_size=args.size)
    
    if args.ic:
        char = allCharacters[random.randint(0, len(allCharacters)-1)]
        chars = [fonts.gen_char_img(char, f_idx, args.font_name)[None, ...] for f_idx in range(num_examples)]
    else:
        chars = [fonts.gen_char_img(allCharacters[random.randint(0, len(allCharacters)-1)], f_idx, args.font_name)[None, ...] for f_idx in range(num_examples)]
        
    chars = torch.from_numpy(np.concatenate(chars)).permute([0, 3, 1, 2])

    save_image(chars, args.save_to, nrow=int(math.sqrt(num_examples)),
            normalize=True, range=(0, 255))