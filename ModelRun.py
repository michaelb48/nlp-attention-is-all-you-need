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


def train_fn(model, dataloader, optimizer, criterion, device, epoch, scheduler, clip=1.0):
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
        scheduler.step()

        # Log progress
        if steps % 100 == 0:
            print(
                f'Epoch: {epoch}, Batch: {steps}, Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.7f}')

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


def train_transformer(model, train_dataloader, val_dataloader, vocab_size, num_epochs,
                      save_path, save_interval,sp, patience=5, avg_n_weights=5,
                      device='cuda'):
    """
    Train the transformer model with validation using perplexity.

    Args:
        model: Transformer model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        vocab_size: Size of vocabulary
        num_epochs: Number of training epochs
        save_path: Path to save model checkpoints
        save_interval: Save model every N iterations
        patience: Number of epochs to wait for improvement before early stopping
        avg_n_weights: average the weights of the model every N iterations
        device: Device to train on
    """

    # these parameters are based on the paper
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=5)  # Ignore padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=0.044, betas=(0.9, 0.98), eps=1e-9)

    # Learning rate scheduler
    def lr_lambda(step):
        warmup_steps = 4000
        step = max(1, step)
        return min(step ** (-0.5), step * warmup_steps ** (-1.5))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Tracking the number of steps and the time during training and evaluation
    global_step = 0
    start_time = time.time()  # Total training start time
    last_save_time = start_time  # Time of the last model save

    # Tracking the best perplexity and bleu score
    best_perplexity = float('inf')
    best_bleu = float('-inf')
    epochs_without_improvement = 0
    best_model_state = None

    # List to store previous state_dict copies for averaging
    prev_state_dicts = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Epoch start time

        # Training phase
        training_perplexity = train_fn(model, train_dataloader, optimizer, criterion, device, epoch, scheduler)

        # Validation phase
        validation_perplexity, validation_bleu = eval_fn(model, val_dataloader, criterion, device, sp)

        print(
            f'Epoch: {epoch}, Train perplexity: {training_perplexity:.4f}, Valid perplexity: {validation_perplexity:.4f}, Valid BLEU4: {validation_bleu:.4f}')

    return model, best_perplexity


if __name__ == '__main__':

    df_corpus = pd.read_pickle('../corpus/df_encoded.pkl')

    vocab = spm.SentencePieceProcessor()
    vocab.load('../bpe/bpe_model.model')
    vocab_size = vocab.get_piece_size()
    sb_vocab = [vocab.id_to_piece(i) for i in range(vocab_size)]
    sb_vocab_dict = {sb_vocab[i]: i for i in range(vocab_size)}

    dataset = TranslationDataset(df_corpus,sb_vocab)

    train_dataloader,val_dataloader = create_train_val_dataloaders(
        dataset,
        batch_size=32,
        vocab=sb_vocab_dict,
        val_split=0.3
    )


    model = Transformer(
        n_vocab_len=vocab_size,
        i_vocab_padding=sb_vocab_dict['<mask>'],
        device='cuda'
    )

    train_transformer(model,train_dataloader,val_dataloader,vocab_size,1,'../model',100, vocab)


