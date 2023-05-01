import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import random
from typing import List
import copy


class SyntaxChecker:
    device: torch.device

    def __init__(self, device):
        self.device = device
        pass

    def reset_state(self, batch_size):
        pass

    def get_syntax_mask(self):
        pass

    def update_state(self, tokens):
        pass



class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, dropout_p: float, padding_tok: int, is_variational: bool = False):
        super().__init__()

        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim, padding_idx=padding_tok)
        
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hid_dim, batch_first=True)

        self.is_variational = is_variational

        if self.is_variational:
            self.mu = nn.Linear(hid_dim, hid_dim)
            self.log_var = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(p=dropout_p)

        
    def forward(self, src):
        #src = [batch_size, seq len]
        
        embedded = self.dropout(self.embedding(src))
        #embedded = [batch size, seq len, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        #outputs = [batch size, seq len, hid dim]
        #hidden = [n layers (1), batch size, hid dim]
        #outputs are always from the top hidden layer

        mu = torch.zeros((src.shape[0], self.hid_dim))
        log_var = torch.zeros((src.shape[0], self.hid_dim))
        if self.is_variational:
            mu = self.mu(hidden.squeeze(dim=0))
            log_var = self.log_var(hidden.squeeze(dim=0))
            #mu =      [batch size, hid dim]
            #log_var = [batch size, hid dim]

            sampled_latent = self.reparameterize(mu, log_var)
            #sampled_latent = [batch size, hid dim]

            hidden = sampled_latent.unsqueeze(0)
            #hidden = [1, batch size, hid dim]
        
        return hidden, mu, log_var
    

    #mu = [batch size, hid dim]
    #log_var = [batch size, hid dim]
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std  # Sample from the Gaussian distribution
        #return = [batch size, hid dim]


class Decoder(nn.Module):

    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, dropout_p: float, padding_tok: int, syntax_checker: SyntaxChecker | None = None):
        super().__init__()

        self.syntax_checker = syntax_checker

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim, padding_idx=padding_tok)
        
        self.rnn = nn.GRU(input_size=emb_dim + hid_dim, hidden_size=hid_dim, batch_first=True)
        
        self.fc_out = nn.Linear(in_features=emb_dim + hid_dim * 2, out_features=output_dim)
        
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, input, hidden, context):
        #input = [batch size]
        #hidden = [n layers (1), batch size, hid dim]
        #context = [n layers (1), batch size, hid dim]
        
        input = input.unsqueeze(1)
        #input = [batch size, 1]
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [batch size, 1, emb dim]
                
        # This is needed to perform the concatenation with the embeddings
        batch_first_context = context.permute(1, 0, 2)
        #batch_first_context = [batch_size, n layers (1), hid dim]

        emb_con = torch.cat((embedded, batch_first_context), dim = 2)
        #emb_con = [batch size, 1, emb dim + hid dim]
            
        output, hidden = self.rnn(emb_con, hidden)
        #output = [batch size, 1, hid dim]
        #hidden = [n layers (1), batch size, hid dim]
        
        output = torch.cat((embedded.squeeze(1), hidden.squeeze(0), context.squeeze(0)), dim = 1)
        #output = [batch size, emb dim + hid dim * 2]
        
        prediction = self.fc_out(output)
        #prediction = [batch size, output_dim]
        
        return prediction, hidden


    def beam_search(self, context, beam_width=5, max_seq_len=100, sos_token=1, eos_token=2) -> List[int]:
        # context = [hid dim]

        context = context.unsqueeze(0).unsqueeze(0)
        # context = [1, 1, hid dim] for [n layers, batch size, hid dim]

        hidden = context
        # hidden = [1, 1, hid dim]

        input = torch.tensor([sos_token]).to(context.device)
        # input = [1] for [batch size]

        if self.syntax_checker:
            self.syntax_checker.reset_state(1)
        sequences = [(input, hidden, 0.0, [], self.syntax_checker)]
        finished_sequences = []
        
        for t in range(1, max_seq_len):
            candidates = []

            for (input, hidden, score, sequence, syntax_checker) in sequences:
                # If the last token in the sequence is the end-of-sentence token, add the sequence to the finished_sequences list
                if input.item() == eos_token:
                    finished_sequences.append((score, sequence))
                    continue

                #insert input token embedding, previous hidden state and the context state
                #receive output tensor (predictions) and new hidden state
                prediction, hidden = self.forward(input, hidden, context)

                if syntax_checker:
                    prediction = prediction + syntax_checker.get_syntax_mask()

                token_log_probs = torch.log_softmax(prediction, dim=1)

                # Get the top 'beam_width' tokens and their corresponding log probabilities
                top_log_probs, top_tokens = torch.topk(token_log_probs, beam_width, dim=1)
                #print(top_probs)

                # Add each of the top tokens as a new sequence candidate
                for idx in range(beam_width):
                    next_input = top_tokens[0][idx].unsqueeze(0)
                    next_hidden = hidden.clone()
                    next_score = score + top_log_probs[0][idx].item()
                    next_sequence = sequence + [next_input.item()]

                    next_syntax_checker = None
                    if syntax_checker:
                        next_syntax_checker = copy.deepcopy(syntax_checker)
                        next_syntax_checker.update_state(next_input)

                    candidates.append((next_input, next_hidden, next_score, next_sequence, next_syntax_checker))
                
            # Sort the new sequence candidates based on their scores and keep the top 'beam_width' candidates for the next iteration
            sorted_best_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]

            sequences = sorted_best_candidates

        finished_sequences = sorted(finished_sequences, key=lambda x: x[0], reverse=True)

        if len(finished_sequences) == 0:
            return None

        return finished_sequences[0][1]



    


class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device, syntax_checker: SyntaxChecker | None = None):
        super().__init__()
        
        self.syntax_checker = syntax_checker

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, seq, teacher_forcing_ratio: float = 0.5, output_latent_rep: bool = False):
        #seq = [batch size, seq len]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = seq.shape[0]
        seq_len = seq.shape[1]
        vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, seq_len, vocab_size).to(self.device)
        
        #last hidden state of the encoder is the context
        context, mu, log_var = self.encoder(seq)
        
        #context also used as the initial hidden state of the decoder
        hidden = context
        
        #first input to the decoder is the <sos> tokens
        input = seq[:,0]
        
        if self.syntax_checker:
            self.syntax_checker.reset_state(batch_size)

        for t in range(1, seq_len):
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

            if self.syntax_checker:
                output = output + self.syntax_checker.get_syntax_mask()
            
            #place predictions in a tensor holding predictions for each token
            outputs[:,t,:] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = seq[:,t] if teacher_force else top1

            if self.syntax_checker:
                self.syntax_checker.update_state(input)

        return outputs, context, mu, log_var
    

class AutoencoderConfig:
    vocab_size: int
    encoder_embed_dim: int
    decoder_embed_dim: int
    latent_rep_dim: int
    encoder_dropout: float
    decoder_dropout: float
    pad_token: int
    syntax_checker: SyntaxChecker | None
    is_variational: bool

    def __init__(
            self,
            vocab_size: int,
            encoder_embed_dim: int,
            decoder_embed_dim: int,
            latent_rep_dim: int,
            encoder_dropout: float,
            decoder_dropout: float,
            pad_token: int,
            syntax_checker: SyntaxChecker | None,
            is_variational: bool
        ):
        self.vocab_size = vocab_size
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.latent_rep_dim = latent_rep_dim
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.pad_token = pad_token
        self.syntax_checker = syntax_checker
        self.is_variational = is_variational
                

def autoencoder_from_config(cfg: AutoencoderConfig, device: torch.device) -> Autoencoder:
    enc = Encoder(cfg.vocab_size, cfg.encoder_embed_dim, cfg.latent_rep_dim, cfg.encoder_dropout, cfg.pad_token, is_variational=cfg.is_variational)
    dec = Decoder(cfg.vocab_size, cfg.decoder_embed_dim, cfg.latent_rep_dim, cfg.decoder_dropout, cfg.pad_token, cfg.syntax_checker)
    autoenc = Autoencoder(enc, dec, device, cfg.syntax_checker).to(device)
    return autoenc

def load_autoencoder(model_path: str, cfg: AutoencoderConfig, device: torch.device) -> Autoencoder:
    autoenc = autoencoder_from_config(cfg, device)
    autoenc.load_state_dict(torch.load(model_path))
    return autoenc