## Here we will be importing a dataset from the Hugging Face library that will be a bilingual dataset.

import torch
import torch.nn
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, source_tokenizer, target_tokenizer, src_lang, tgt_lang, seq_len):
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        ## converting the sos and eos tokens into token ids
        self.sos_token_id = torch.tensor([self.target_tokenizer.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token_id = torch.tensor([self.target_tokenizer.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token_id = torch.tensor([self.target_tokenizer.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        source = self.source_tokenizer.encode(example['translation'][self.src_lang]).ids
        target = self.target_tokenizer.encode(example['translation'][self.tgt_lang]).ids

        ##Now the source and target are the token ids as we have encoided them using the tokenizers
        encoding_paddingTokens = self.seq_len - len(source) -2 #(we have the sos and eos)
        decoding_paddingTokens = self.seq_len - len(target) -1 #(we have the sos and eos) while training we only add start of the sentence token and hence we subtract 1. We add end of the sentence token to the label

        if encoding_paddingTokens < 0 or decoding_paddingTokens < 0:
            raise ValueError('The sequence length is too long.')
        
        encoder_input = torch.cat(self.sos_token_id, torch.tensor(source, dtype=torch.int64),
                                  self.eos_token_id, 
                                  torch.tensor([self.pad_token_id]*encoding_paddingTokens))
        
        decoder_input = torch.cat(self.sos_token_id, torch.tensor(target, dtype=torch.int64),
                                    torch.tensor([self.pad_token_id]*decoding_paddingTokens))
        
        label = torch.cat(torch.tensor(target, dtype=torch.int64), self.eos_token_id,
                            torch.tensor([self.pad_token_id]*decoding_paddingTokens))
        
        ## we make sure that the length of the input and the label is equal to the sequence length
        assert len(encoder_input) == self.seq_len
        assert len(decoder_input) == self.seq_len
        assert len(label) == self.seq_len
        

        return {
            'enc_input': encoder_input,
            'dec_input ': decoder_input,
            'label': label,
            'encoder_mask': (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int(), ## we dont want padding tokens to participate in self attention, we also add the seq and the batch dimeniions
            ## the shape of the encoder input will be (1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)).int() 
            ## we dont want padding tokens to participate in self attention, we also add the seq and the batch dimensions. It will have (1,seq_len,seq_len) shape
        }
    
def causal_mask(seq_len):
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).torch,type(torch.int64)
    return mask == 0