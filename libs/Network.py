import torch.nn as nn

class TransformerMultiFeature(nn.Module):
    def __init__(self, num_features, d_model=64, num_layers=2, nhead=4, dim_feedforward=128):
        super(TransformerMultiFeature, self).__init__()
        self.num_features = num_features

        # Embedding layers
        self.input_embedding = nn.Linear(num_features, d_model)
        self.output_embedding = nn.Linear(num_features, d_model)

        # Transformer components
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, num_features)

    def forward(self, src, tgt):
        # Embed inputs
        src = self.input_embedding(src)  # Shape: (batch_size, input_window, d_model)
        tgt = self.output_embedding(tgt)  # Shape: (batch_size, output_window, d_model)

        # Permute for Transformer (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer encoding-decoding
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)

        # Final output
        output = self.fc_out(output)  # Shape: (output_window, batch_size, num_features)

        # Permute back to (batch_size, output_window, num_features)
        return output.permute(1, 0, 2)
