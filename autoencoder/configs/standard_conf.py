from autoencoder.autoencoder_model import AutoencoderConfig, SyntaxChecker

standard_autoencoder_config = AutoencoderConfig(
	vocab_size=11,
	encoder_embed_dim=256,
    decoder_embed_dim=256,
    latent_rep_dim=256,
    encoder_dropout=0.3,
    decoder_dropout=0.3,
    pad_token=0,
    syntax_checker=None,
    is_variational=False
)