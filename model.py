# !pip install torch
import torch
import torch.nn as nn
import math 

# import math
class inputembeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(inputembeddings, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding  = nn.Embedding(vocab_size, d_model) ## how many words in the vocab, so d_model is the dimensions of the embeddings - 512 in this case 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) ## multiply by sqrt(d_model) as per the paper
    
class positionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length, dropout ) -> None :
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        ## create a matrix of shape seq_length x d_model
        pe = torch.zeros(seq_length, d_model) 

        ## create a position vector
        position = torch.arange(0, seq_length).unsqueeze(1).float()

        ## create a divison factor
        division_term = torch.exp(torch.arange(0, d_model,2).float()) * (-math.log(10000.0)/d_model) ## we have taken minus because we will be multiplying

        ## apply sin to the even postions and cos to the odd positions. This is done for each of the words in the sentence.

        pe[:, 0::2] = torch.sin(position * division_term)
        pe[:, 1::2] = torch.cos(position * division_term)

        pe = pe.unsqueeze(0) ## add a batch dimension to the positional encoding

        self.register_buffer('pe', pe) ## register the positional encoding as a buffer so that it is not updated during training

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) ## add the positional encoding to the input embeddings, 
        #also we do requires_grad as false so that the positional encoding is not updated during training or not leaned during training



class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model)) ## multiplying by ones to keep the original value
        self.beta = nn.Parameter(torch.zeros(d_model)) ##adding zeros to keep the original value

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(FeedForwardLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x =self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = heads
        self.d_k = d_model // heads
        self.dropout = nn.Dropout(dropout)

        ## defining the query, key and value matrices
        self.w_query = nn.Linear(d_model, d_model)  ## this is the Wq matrix - this is the matrix that will be used to calculate the query
        self.w_key = nn.Linear(d_model, d_model)  ## this is the Wk matrix - this is the matrix that will be used to calculate the key
        self.w_value = nn.Linear(d_model, d_model)  ## this is the Wv matrix  - this is the matrix that will be used to calculate the value 

        self.w_o = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout ):
        d_k = query.size(-1)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) ## we are dividing by sqrt(d_k) as per the paper
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) ## (batch_size, h, seq_length, seq_length)

        if dropout:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, value), attention_scores



        ## we call the matrices by nn.Linear as it does a linear transformation of the input data

    def forward(self, query, key, value, mask):
        query = self.w_query(query) ##(batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model) , dimensions remains the same as we are just doing a linear transformation
        key = self.w_key(key) ##(batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        value = self.w_value(value) ##(batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)

        ## we now split the query, key and value into h heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).permute(0, 2, 1, 3) ## (batch_size, seq_length, d_model) --> (batch_size, h, seq_length, d_k)
        ## so basicallyy each head sees the full sentence but only a part of the dimensions
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).permute(0, 2, 1, 3)

        prod, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout) ## (batch_size, h, seq_length, d_k)
        prod = prod.transpose(1, 2).contiguous().view(prod.shape[0], -1, self.h * self.d_k) ## (batch_size, seq_length, h, d_k) --> (batch_size, seq_length, d_model)

        return self.w_o(prod) ## (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)



## Here we will code up the residual connection and the layer normalization

class residualConnection(nn.Module):
    def __init__(self, dropout):
        super(residualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardLayer, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([residualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask): ## src_mask is the mask that is used to mask out the padding tokens
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)


class DecodderBlock(nn.Module):
    def __init__(self, dropout: float, self_attention_block = MultiHeadAttention, cross_attention_block = MultiHeadAttention, feedforwardblock = FeedForwardLayer) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforwardblock = feedforwardblock
        self.residual_connections = nn.ModuleList([residualConnection(dropout) for i in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask): # source mask comes from the encoder and tgt_mask comes from the decoder. The source is will be the source language and the target will be the target language as this a translation task
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feedforwardblock)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for l in self.layers:
            x = l(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


 ## this layer converts the embedding to the vocabulary
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim = -1) ## we take the log softmax as we are using the negative log likelihood loss function    
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder,src_embed: inputembeddings, tgt_embed: inputembeddings, src_pos: positionalEncoding, tgt_pos: positionalEncoding, proj=ProjectionLayer ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask) 
    
    def project(self, x):
        return self.proj(x)
    

def build_transformer(src_vocab:int, tgt_vocab_size:int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N:int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    ## create embeddings for the source and target languages
    src_embed = inputembeddings(src_vocab, d_model)
    tgt_embed = inputembeddings(tgt_vocab_size, d_model)

    ## create positional encodings for the source and target languages
    src_pos = positionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = positionalEncoding(d_model, tgt_seq_len, dropout)

    # create encoder blocks
    encoder_blocks = []

    for i in range(N):
        encoder_blocks.append(EncoderBlock(MultiHeadAttention(d_model, h, dropout), FeedForwardLayer(d_model, d_ff, dropout), dropout))

    ## create decoder blocks
    decoder_blocks = []

    for i in range(N):
        decoder_blocks.append(DecodderBlock(MultiHeadAttention(d_model, h, dropout), MultiHeadAttention(d_model, h, dropout), FeedForwardLayer(d_model, d_ff, dropout), dropout))

    ## create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    ## create the projection layer
    proj = ProjectionLayer(d_model, tgt_vocab_size)

    ## create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj)

    ## Inititalize the weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer