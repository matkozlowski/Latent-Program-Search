from autoencoder.autoencoder_model import Autoencoder
import torch
from typing import List, Any
from tqdm import tqdm



def get_perfect_reconstruction_accuracy(model: Autoencoder, test_set: List[torch.Tensor], beam_width: int):
    model.eval()

    device = model.device

    total_perfect = 0
    total_wrong = 0
    
    for i, item in enumerate(pbar := tqdm((test_set))):
        latent_rep, _, _ = model.encoder(item.unsqueeze(0))
        latent_rep = latent_rep.squeeze()
        output = model.decoder.beam_search(latent_rep, beam_width=beam_width)
        
        target = item.tolist()[1:]
        
        if target == output:
            total_perfect += 1
        else:
            total_wrong += 1
            print(target)
            print(output)

        pbar.set_description(f'Percent Perfect: {total_perfect / (i+1) * 100:.3f}%')
    
    print(f'Got {total_wrong} wrong')

