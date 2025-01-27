import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import sentencepiece as spm
from Transformer import Transformer
from TranslationDataset import TranslationDataset, create_train_val_dataloaders
from torchtext.data.metrics import bleu_score

def print_training_parameters(num_epochs, save_path, save_interval, optimizer, criterion):
    print(f"""
    Training Parameters:
    ---------------------
    Number of Epochs   : {num_epochs}
    Save Path          : {save_path}
    Save Interval      : {save_interval}
    Optimizer          : NoamOptim(
        optimizer: torch.optim.Adam(
            lr={optimizer.optimizer.param_groups[0]['lr']},
            betas={optimizer.optimizer.param_groups[0]['betas']},
            eps={optimizer.optimizer.param_groups[0]['eps']}
        ),
        d_model={optimizer.d_model},
        warmup_steps={optimizer.n_warmup_steps},
        factor={optimizer.factor}
    )
    Criterion          : nn.CrossEntropyLoss(
        ignore_index={criterion.ignore_index}
    )
    """)
    
def train_fn(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    steps = 0
    tk0 = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    output = None
    
    for batch in tk0:

        source = batch[0].to(device)
        target = batch[1].to(device)

        # forward pass
        optimizer.zero_grad()
        output = model(source, target[:, :-1])

        # calculate the loss
        loss = criterion(
            output.view(-1, output.size(-1)),  # (batch_size * (target_seq_len - 1), vocab_size)
            target[:, 1:].contiguous().view(-1)  # (batch_size * (target_seq_len - 1))
        )

        total_loss += loss.item()
        steps += 1

        output = output.argmax(dim=-1)
        # backward pass
        loss.backward()
        # clip gradients to avoid exploding gradients issue
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        # update model parameters
        optimizer.step()
        #scheduler.step()

        # Log progress
        if steps % 100 == 0:
            print(
                f'Batch: {steps}, Loss: {loss.item():.4f}, Learning Rate: {optimizer.get_lr()[0]:.7f}')

        tk0.set_postfix(loss=total_loss / steps)
    tk0.close()
    perplexity = np.exp(total_loss / len(dataloader))

    return perplexity


def eval_fn(model, dataloader, criterion, device, sp):
    model.eval()
    total_loss = 0.0
    steps = 0
    hypotheses = []
    references = []

    # Load the BLEU metric
    # bleu = load_metric("bleu")

    tk0 = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    with torch.no_grad():
        for batch in tk0:
            source = batch[0].to(device)
            target = batch[1].to(device)

            # forward pass
            optimizer.zero_grad()
            output = model(source, target[:, :-1])

            # calculate the loss
            loss = criterion(
                output.view(-1, output.size(-1)),  # (batch_size * (target_seq_len - 1), vocab_size)
                target[:, 1:].contiguous().view(-1)  # (batch_size * (target_seq_len - 1))
            )

            total_loss += loss.item()
            steps += 1
            output = output.argmax(dim=-1)
            target = target[:, 1:]

            # converting the ids to tokens for bleu score
            # pred_tokens = convert_ids_to_text(output, de_text.vocab, EOS_IDX, UNK_IDX)
            # target_tokens = convert_ids_to_text(target, de_text.vocab, EOS_IDX, UNK_IDX)
            pred_tokens = sp.encode_as_pieces(sp.decode(output[0].cpu().tolist()))
            target_tokens = sp.encode_as_pieces(sp.decode(target[0].cpu().tolist()))
            print("Expected Output:", target_tokens)
            print("Predicted Output:", pred_tokens)
            hypotheses += pred_tokens
            references += [[token] for token in target_tokens if token != '<mask>']
            tk0.set_postfix(loss=total_loss / steps)
    tk0.close()
    perplexity = np.exp(total_loss / len(dataloader))
    references = [[[item[0] for item in references]]]
    hypotheses = [hypotheses]
    # print(f"hypotheses: {hypotheses}")
    # print(f"references: {references}")
    # Compute the BLEU score
    bleu_score = 0 # bleu.compute(predictions=hypotheses, references=references)

    return perplexity, bleu_score

class NoamOptim(object):
    """ Optimizer wrapper for learning rate scheduling.
    """

    def __init__(self, optimizer, d_model, factor, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
    

    def zero_grad(self):
        self.optimizer.zero_grad()


    def step(self):
        self.n_steps += 1
        lr = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

    
    def get_lr(self):
        return self.factor * (
            self.d_model ** (-0.5)
            * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))
        )

def train_transformer(model, train_dataloader, val_dataloader, num_epochs,
                      save_path, save_interval, optimizer, criterion, sp, es_patience=5, avg_n_weights=5,
                      device='cuda'):
    
    best_bleu4 = float('-inf')
    patience = 0
    N_EPOCHS = num_epochs
    CLIP = 1.0
    
    for epoch in range(0, N_EPOCHS + 1):
        # one epoch training

        train_perplexity = train_fn(model, train_dataloader, optimizer, criterion, device, CLIP)
        
        # one epoch validation
        valid_perplexity, valid_bleu4 = eval_fn(model, val_dataloader, criterion, device, sp)
        
        print(f'Epoch: {epoch}, Train perplexity: {train_perplexity:.4f}, Valid perplexity: {valid_perplexity:.4f}, Valid BLEU4: {valid_bleu4:.4f}')
        
        # early stopping
        is_best = valid_bleu4 > best_bleu4
        if is_best:
            print(f'BLEU score improved ({best_bleu4:.4f} -> {valid_bleu4:.4f}). Saving Model!')
            best_bleu4 = valid_bleu4
            patience = 0
            torch.save(model.state_dict(), save_path + f'/model_{epoch}.pth')
        else:
            patience += 1
            print(f'Early stopping counter: {patience} out of {es_patience}')
            if patience == es_patience:
                print(f'Early stopping! Best BLEU4: {best_bleu4:.4f}')
                break

    return model

if __name__ == '__main__':
    corpus_path = '../corpus/df_encoded.pkl'
    print(f"Loading corpus from: {corpus_path} ...")
    df_corpus = pd.read_pickle(corpus_path)

    bpe_model_path = '../bpe/bpe_model.model'
    print(f"Loading BPE model from: {bpe_model_path} ...")
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path)
    sb_vocab_size = sp.get_piece_size()
    sb_vocab = [sp.id_to_piece(i) for i in range(sb_vocab_size)]
    sb_vocab_dict = {sb_vocab[i]: i for i in range(sb_vocab_size)}

    print("Creating dataset ...")
    dataset = TranslationDataset(df_corpus, sb_vocab)

    print("Creating data loaders ...")
    train_dataloader,val_dataloader = create_train_val_dataloaders(
        dataset,
        batch_size=32,
        vocab=sb_vocab_dict,
        val_split=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Initializing model ...")
    model = Transformer(
        n_vocab_len=sb_vocab_size,
        i_vocab_padding=sb_vocab_dict['<mask>'],
        device=device
    ).to(device)

    num_epochs = 10
    save_path = 'models'
    save_interval = None
    optimizer = NoamOptim(
        torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9),
        model.d_model, 2, 4000
    )
    criterion = nn.CrossEntropyLoss(ignore_index=sb_vocab_dict['<mask>'])
    
    print_training_parameters(
        num_epochs=10,
        save_path='models',
        save_interval=None,
        optimizer=NoamOptim(
            torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9),
            model.d_model, 2, 4000
        ),
        criterion=nn.CrossEntropyLoss(ignore_index=sb_vocab_dict['<mask>'])
    )

    print("Starting training!")
    train_transformer(model, train_dataloader, val_dataloader, num_epochs,
                     save_path, save_interval, optimizer, criterion, sp, es_patience=5, avg_n_weights=5,
                     device='cuda')

