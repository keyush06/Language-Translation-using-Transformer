import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace 
from dataset import BilingualDataset, causal_mask
from model import build_transformer
import warnings
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path

from pathlib import Path

def get_all_sentences(dataset, language):
    for example in dataset:
        yield example['translation'][language]

def get_or_create_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))

    if not Path.exists (tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]", "[MASK]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer)

        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataloaders(config):
    dataset = load_dataset('opus_books', f'{config["source_language"]}-{config["target_language"]}', split='train')

    source_tokenizer = get_or_create_tokenizer(config, dataset, config['source_language'])
    target_tokenizer = get_or_create_tokenizer(config, dataset, config['target_language'])

    ## Kepp 80% for training and 20% for validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, val_size])

    train_dataset = BilingualDataset(train_dataset, source_tokenizer, target_tokenizer, config['source_language'], config['target_language'], config['seq_len'])
    test_dataset = BilingualDataset(test_dataset, source_tokenizer, target_tokenizer, config['source_language'], config['target_language'], config['seq_len'])

    max_len_src, max_len_tgt = 0, 0

    for item in dataset:
        src_ids = source_tokenizer.encode(item['translation'][config['source_language']]).ids
        tgt_ids = target_tokenizer.encode(item['translation'][config['target_language']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Maximum source length: {max_len_src}')
    print(f'Maximum target length: {max_len_tgt}')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, source_tokenizer, target_tokenizer

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(config, vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    ## define the device
    device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ## creating the weights folder
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    ## get the dataloaders
    train_loader, test_loader, source_tokenizer, target_tokenizer = get_dataloaders(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size())

    ##tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimier = torch.optim.Adam(model.parameters(), lr=config['lr'])

    ##state of the model and optimzier is resotred from the latest weights file if the training crashes or is stopped
    initital_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Loading model from {model_filename}')
        state = torch.load(model_filename)
        initital_epoch = state['epoch'] + 1
        optimier.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(initital_epoch, config['num_epochs']):
        model.train() ## set the model to training mode
        batch_iter = tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch')
        for batch in batch_iter:
            encoder_inp = batch['encoder_input'].to(device) ## (batch_size, seq_len)
            decoder_inp = batch['decoder_input'].to(device) ## (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) ## (batch_size, 1,1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) ## (batch_size, 1, seq_len, seq_len) ## here we hide the words that the model should not see


            ## run the tensors theough the transformer
            encoder_output = model.encoder(encoder_inp, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decoder(decoder_inp, encoder_output, encoder_mask, decoder_mask)# (batch_size, seq_len, d_model)
            ## Now we have the projection layer for projecting it to the vocab size
            projected_output = model.project(decoder_output) ## (batch_size, seq_len, vocab_size)

            ## extracting the target labels
            labels = batch['label'].to(device) ## (batch_size, seq_len)

            ## compuiting the loss
            loss  = loss_fn(projected_output.view(-1, target_tokenizer.get_vocab_size(-1)), labels.view(-1))
            batch_iter.set_postfix({f'loss = ': f'{loss.item():6.3f}'})

            ##log the loss
            writer.add_scalar('loss', loss.item(), global_step)
            writer.flush()

            ##backpropagation
            loss.backward()

            ## update the weights of the model
            optimier.step()
            optimier.zero_grad()

            global_step += 1

            ## save the model after each epoch
            model_filename = get_weights_file_path(config, str(epoch))
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimier.state_dict()
            }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)

