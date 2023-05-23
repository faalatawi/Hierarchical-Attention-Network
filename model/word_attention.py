import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class myWordAttention(nn.Module):

    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()


        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.gru = nn.GRU(
            embed_dim,
            gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * gru_hidden_dim, att_dim) # FIXME -> nn.LayerNorm((2 * gru_hidden_dim, att_dim))

        self.dropout = nn.Dropout(dropout)

        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)

        self.context_vector = nn.Linear(att_dim, 1, bias=False)

        def init_embeddings(self, embeddings):
            self.embeddings.weight = nn.Parameter(embeddings)

        def freeze_embeddings(self, freeze):
            self.embeddings.weight.requires_grad = not freeze

        def forward(self, sents, sent_lengths):

            sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)

            sents = sents[sent_perm_idx]


            sents = self.embeddings(sents)
            sents = self.dropout(sents)

            packed_words : PackedSequence = pack_padded_sequence(sents, lengths= sent_lengths.tolist(), batch_first=True)

            valid_bsz = packed_words.batch_sizes


            packed_words, _ = self.gru(packed_words)

            if self.use_layer_norm:
                normed_words = self.layer_norm(packed_words.data)

            else:
                normed_words = packed_words


            # word attention
            att = torch.tanh(self.attention(normed_words.data))
            att = self.context_vector(att).squeeze(1)

            val = att.max()
            att = torch.exp(att - val)


            att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first= True)

            att_weights = att / torch.sum(att, dim = 1, keepdim = True)

            # Compute sentence vectors
            sents = sents * att_weights.unsqueeze(2)
            sents = sents.sum(dim=1)



            sents, _ = pad_packed_sequence(packed_words, batch_first=True)

            _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
            sents = sents[sent_unperm_idx]

            # NOTE MODIFICATION BUG
            att_weights = att_weights[sent_unperm_idx]

            return sents, att_weights
