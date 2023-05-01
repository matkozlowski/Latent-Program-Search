from autoencoder.ast_dataset import ASTDataset, SameLengthBatchSampler
from autoencoder.autoencoder_model import Encoder, Decoder, Autoencoder, AutoencoderConfig, autoencoder_from_config, SyntaxChecker
from autoencoder.syntax_checkers.arithmetic_syntax_checker import ArithmeticSyntaxChecker
from autoencoder.semantic_ladder_loss import SemanticLadderLoss, SemanticLadderLossHelper
from grammars.arithmetic_grammar import ArithmeticGrammar
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from tqdm import tqdm
import math
from typing import List



def train_epoch(model: Autoencoder, dataloader, optimizer, loss_func, clip, semantic_loss_weight: float = 0, kld_loss_weight: float = 0, semantic_loss_func=None):
    
    model.train()
    
    epoch_loss = 0
    epoch_reconstruction_loss = 0
    epoch_semantic_loss = 0
    epoch_kld_loss = 0
    
    for x in tqdm(dataloader):
        optimizer.zero_grad()


        output, latent_rep, mu, log_var = model(x)
        #x = [batch size, seq len]
        #output = [batch size, seq len, vocab size]
        #latent_rep = [1, batch_size, hid dim]
        
        output_dim = output.shape[-1]
        
        output = output[:,1:].reshape(-1, output_dim)
        y = x[:,1:].reshape(-1)
        #y = [(seq len - 1) * batch size]
        #output = [(seq len - 1) * batch size, output dim]

        reconstruction_loss = loss_func(output, y)
        
        semantic_loss = 0
        if semantic_loss_func is not None:
            if model.encoder.is_variational:
                semantic_loss = semantic_loss_func(x, mu, model.encoder, model.device, use_mu=True)
            else:
                semantic_loss = semantic_loss_func(x, latent_rep, model.encoder, model.device)

        kld_loss = 0
        if model.encoder.is_variational:
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconstruction_loss + semantic_loss_weight * semantic_loss + kld_loss * kld_loss_weight
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_reconstruction_loss += reconstruction_loss
        epoch_semantic_loss += semantic_loss * semantic_loss_weight
        epoch_kld_loss += kld_loss * kld_loss_weight
        
    return epoch_loss / len(dataloader), epoch_reconstruction_loss / len(dataloader), epoch_semantic_loss / len(dataloader), epoch_kld_loss / len(dataloader)


def evaluate_epoch(model: Autoencoder, dataloader, loss_func):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for x in dataloader:
            output, _, _, _ = model(x, 0) #turn off teacher forcing

            output_dim = output.shape[-1]
            
            output = output[:,1:].reshape(-1, output_dim)
            y = x[:,1:].reshape(-1)

            loss = loss_func(output, y)

            epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(n_epochs, save_path, model_config: AutoencoderConfig, train_data_path, valid_data_path, batch_size=32, 
          syntax_checker: SyntaxChecker | None = None, use_semantic_ladder_loss: bool = False, 
          use_variational: bool = False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLIP = 1

    if syntax_checker is not None:
        model_config.syntax_checker = syntax_checker

    train_dataset = ASTDataset(train_data_path, model_config.pad_token, device)
    valid_dataset = ASTDataset(valid_data_path, model_config.pad_token, device)
    train_sampler = SameLengthBatchSampler(train_dataset, batch_size)
    valid_sampler = SameLengthBatchSampler(valid_dataset, batch_size)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_sampler=valid_sampler)

    KLD_LOSS_WEIGHT = 10
    SEMANTIC_LOSS_WEIGHT = 1

    # Create semantic ladder loss functions if specified
    train_semantic_ladder_loss = None
    valid_semantic_ladder_loss = None
    if use_semantic_ladder_loss:
        LADDER_LEVELS = 10
        MARGIN = 0.05
        LADDER_WEIGHT = 1.0
        semantic_loss_margins = [0.2, 0.15, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        semantic_loss_ladder_weights = [1, 0.5, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        grammar = ArithmeticGrammar()
        train_helper = SemanticLadderLossHelper('autoencoder/data/semantic_loss_data/train_behavior_dist_mat.npy', 
                                                'autoencoder/data/semantic_loss_data/train_behavior_vecs.npy', 
                                                'autoencoder/data/semantic_loss_data/train_behavior_dist_mat_idx.pkl',
                                                train_dataset, inputs=None, grammar=grammar)
        train_semantic_ladder_loss = SemanticLadderLoss(margins=semantic_loss_margins, ladder_levels=LADDER_LEVELS,
                                                        ladder_weights=semantic_loss_ladder_weights, helper=train_helper)
        
        valid_helper = SemanticLadderLossHelper('autoencoder/data/semantic_loss_data/valid_behavior_dist_mat.npy', 
                                                'autoencoder/data/semantic_loss_data/valid_behavior_vecs.npy', 
                                                'autoencoder/data/semantic_loss_data/valid_behavior_dist_mat_idx.pkl',
                                                valid_dataset, inputs=None, grammar=grammar)
        valid_semantic_ladder_loss = SemanticLadderLoss(margins=semantic_loss_margins, ladder_levels=LADDER_LEVELS,
                                                        ladder_weights=semantic_loss_ladder_weights, helper=valid_helper)         

    if use_variational:
        model_config.is_variational = True

    model = autoencoder_from_config(model_config, device)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())

    loss_func = nn.CrossEntropyLoss(ignore_index=model_config.pad_token)

    best_val_loss = 10000
    for epoch in range(n_epochs):
        train_loss, train_recon_loss, train_sem_loss, train_kld_loss = train_epoch(model, train_dataloader, optimizer, loss_func, semantic_loss_weight=SEMANTIC_LOSS_WEIGHT, clip=CLIP, semantic_loss_func=train_semantic_ladder_loss, kld_loss_weight=KLD_LOSS_WEIGHT)
        valid_loss = evaluate_epoch(model, valid_dataloader, loss_func)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), save_path)

        print(f'EPOCH {epoch}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tTrain Reconstruction Loss: {train_recon_loss:.3f} | Train Reconstruction PPL: {math.exp(train_recon_loss):7.3f}')
        print(f'\tTrain Semantic Loss: {train_sem_loss:.3f} | Train Semantic PPL: {math.exp(train_sem_loss):7.3f}')
        print(f'\tTrain KL Divergence Loss: {train_kld_loss:.3f} | Train KLD PPL: {math.exp(train_kld_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}\n')
