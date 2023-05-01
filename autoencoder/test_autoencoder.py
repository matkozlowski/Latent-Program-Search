from autoencoder_model import Encoder, Decoder, Autoencoder
import torch
from torch.nn.functional import cosine_similarity
from syntax_checkers.arithmetic_syntax_checker import ArithmeticSyntaxChecker


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PAD_TOKEN = 0
    vocab_size = 10
    INPUT_DIM = vocab_size
    OUTPUT_DIM = vocab_size
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 256
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    MODEL_SAVE_PATH = 'best_model.pt'

    syntax_checker = ArithmeticSyntaxChecker(device, max_depth=3, inf_mask=True)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT, PAD_TOKEN)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, PAD_TOKEN, syntax_checker) 
    model = Autoencoder(enc, dec, device, syntax_checker).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    
    # test_input = torch.tensor([[1, 5, 9, 5, 7, 4, 7, 7, 2]]).to(device)
    # test_latent_rep_1 = model.encoder(test_input).squeeze()
    
    # test_input = torch.tensor([[1, 5, 9, 5, 9, 4, 7, 7, 2]]).to(device)
    # test_latent_rep_2 = model.encoder(test_input).squeeze()

    # test_input = torch.tensor([[1, 5, 9, 7, 2]]).to(device)
    # test_latent_rep_3 = model.encoder(test_input).squeeze()

    # print(torch.sqrt(torch.sum((test_latent_rep_1 - test_latent_rep_2) ** 2)))
    # print(torch.sqrt(torch.sum((test_latent_rep_1 - test_latent_rep_3) ** 2)))

    # print(1 - cosine_similarity(test_latent_rep_1.unsqueeze(0), test_latent_rep_2.unsqueeze(0)))
    # print(1 - cosine_similarity(test_latent_rep_1.unsqueeze(0), test_latent_rep_3.unsqueeze(0)))


    test_input = torch.tensor([[1, 7, 2]]).to(device)
    test_latent_rep = model.encoder(test_input).squeeze()

    orig_seq = model.decoder.beam_search(test_latent_rep, beam_width=1)
    print(orig_seq)
    test_latent_rep_orig = test_latent_rep.clone()
    for lr_mod in range(256):
        test_latent_rep = test_latent_rep_orig.clone()
        for i in range(100):
            test_latent_rep[lr_mod] += 1
            #print(test_latent_rep)
            pred_seq = model.decoder.beam_search(test_latent_rep, beam_width=1)
            #print(pred_seq)
            if pred_seq != orig_seq:
                print(pred_seq)
                break
        print(lr_mod)

if __name__ == '__main__':
    main()
