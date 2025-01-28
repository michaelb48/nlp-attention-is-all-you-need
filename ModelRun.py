import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm
from Transformer import Transformer
from TranslationDataset import TranslationDataset, create_train_val_dataloaders
from torchtext.data.metrics import bleu_score
from Optimizer import CustomOptim
from itertools import islice
import json


def train_fn(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    steps = 0
    tk0 = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    output = None
    global batch_start
    
    for batch_idx, batch in enumerate(tk0):
        if batch_idx < batch_start:
            print(f"Skipping: {batch_idx}...")
            continue

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
        batch_start += 1

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
                f'Batch: {steps}, Loss: {loss.item():.4f}, Learning Rate: {optimizer.get_lr():.7f}')

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

    # this is the path to the experiment configuration; set the values in the config file to execute a new experiment
    EX_CONFIG_PATH = "/config/ex_config.json"

    # Open and load the JSON file into a dictionary
    with open(EX_CONFIG_PATH, 'r') as file:
        config = json.load(file)

    # VARIABLES FROM CONFIG FILE THAT CONTROL EXPERIMENT RUN
    pytorch_cuda_config = config.get('pytorch_cuda','max_split_size_mb:128')
    
    corpus_path_config = config.get('corpus_path','/corpus/df_encoded.pkl')
    bpe_model_path_config = config.get('bpe_model_path','/bpe/bpe_model.model')
    
    batch_size_config = config.get('batch_size',16)
    dataset_value_split_config = onfig.get('dataset_value_split',0.1)

    lr_config = config.get('lr',1e-4)
    beta1_config = config.get('beta1',0.9)
    beta2_config = config.get('beta2',0.98)
    eps_config = config.get('eps',1e-9)
    warmup_steps_config = config.get('warmup_steps',4000)
    lr_factor_config = config.get('lr_factor',1)

    training_time_in_minutes_config = config.get('training_time_in_minutes', 180)
    training_steps_per_epoch_config = config.get('training_steps_per_epoch', 20000)
    model_save_path_config = config.get('model_save_path','/models')
    save_interval_in_minutes_config = config.get('save_interval_in_minutes',10)
    average_model_weight_num_config = config.get('average_model_weight_num',5)
    
    beam_size_config = config.get('beam_size',4)
    len_penalty_alpha_config = config.get('len_penalty_alpha',0.6)
    max_len_a_config = config.get('max_len_a',1)
    max_len_b_config = config.get('max_len_b',50)

    
    d_model_config = config.get('d_model_config',512)
    
    d_dec_ff_inner_config = config.get('d_dec_ff_inner',2048)
    t_dec_heads_config = config.get('t_dec_heads',8)
    t_dec_layer_num_config = config.get('t_dec_layer_num',6)
    
    d_enc_ff_inner_config = config.get('d_enc_ff_inner',2048)
    t_enc_heads_config = config.get('t_enc_heads',8)
    t_enc_layer_num_config = config.get('t_enc_layer_num',6)
    
    d_query_key_head_config = config.get('d_query_key_head',64)
    d_value_head_config = config.get('d_value_head',64)
    
    t_dropout_config = config.get('t_dropout',0.1)
    t_dot_product_config = config.get('t_dot_product',True)

    beam_size_config = config.get('beam_size',4)
    len_penalty_alpha_config = config.get('len_penalty_alpha','max_split_size_mb:128')
    max_len_a_config = config.get('max_len_a','max_split_size_mb:128')
    max_len_b_config = config.get('max_len_b','max_split_size_mb:128')
    
        
    # INIT EXPERIMENT RUN
    
    # set general training parameters
    training_time_in_minutes = training_time_in_minutes_config
    training_steps_per_epoch = training_steps_per_epoch_config
    model_save_path = model_save_path_config
    save_interval_in_minutes = save_interval_in_minutes_config
    average_model_weight_num = average_model_weight_num_config
    
    # set cuda configuration for experiments
    print(f"Cuda allocation configuration is: {pytorch_cuda_config} ...")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = pytorch_cuda_config

    # load corpus for experiment
    print(f"Loading corpus from: {corpus_path_config} ...")
    df_corpus = pd.read_pickle(corpus_path_config)

    # load bpe model for experiment
    print(f"Loading BPE model from: {bpe_model_path_config} ...")
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path_config)

    # create variables for model from bpe model
    sb_vocab_size = sp.get_piece_size()
    sb_vocab_list = [sp.id_to_piece(i) for i in range(sb_vocab_size)]
    sb_vocab_dict = {sb_vocab_list[i]: i for i in range(sb_vocab_size)}

    # initialize dataset
    print("Creating dataset ...")
    dataset = TranslationDataset(df_corpus, sb_vocab_list)
    print("Creating data loaders ...")
    train_dataloader, val_dataloader = create_train_val_dataloaders(
        dataset,
        batch_size=batch_size_config,
        vocab=sb_vocab_dict,
        val_split=dataset_value_split_config
    )
    
    # set the device for experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device == 'cuda':
        torch.cuda.empty_cache()

    # initialize the model
    print("Initializing model ...")
    model = Transformer(
        n_vocab_len=sb_vocab_size,
        i_vocab_padding=sb_vocab_dict['<mask>'],
        d_model=d_model_config,
        device=device,
        d_dec_ff_inner=d_dec_ff_inner_config,
        t_dec_heads=t_dec_heads_config,
        t_dec_layer_num=t_dec_layer_num_config,
        d_enc_ff_inner=d_enc_ff_inner_config,
        t_enc_heads=t_enc_heads_config, 
        t_enc_layer_num=t_enc_layer_num_config,
        d_query_key_head=d_query_key_head_config,
        d_value_head=d_value_head_config,
        t_dropout=t_dropout_config,
        t_dot_product=t_dot_product_config
    ).to(device)

    d_model_config = config.get('d_model_config',512)
    
    d_dec_ff_inner_config = config.get('d_dec_ff_inner',2048)
    t_dec_heads_config = config.get('t_dec_heads',8)
    t_dec_layer_num_config = config.get('t_dec_layer_num',6)
    
    d_enc_ff_inner_config = config.get('d_enc_ff_inner',2048)
    t_enc_heads_config = config.get('t_enc_heads',8)
    t_enc_layer_num_config = config.get('t_enc_layer_num',6)
    
    d_query_key_head_config = config.get('d_query_key_head',64)
    d_value_head_config = config.get('d_value_head',64)
    
    t_dropout_config = config.get('t_dropout',0.1)
    t_dot_product_config = config.get('t_dot_product',True)

    
    # initialize the optimizer
    print("Initializing optimizer ...")
    optimizer = CustomOptim(
        optimizer=torch.optim.Adam(model.parameters(), lr=lr_config, betas=(beta1_config, beta2_config), eps=eps_config),
        lr=lr_config,
        beta1 = beta1_config,
        beta2 = beta2_config,
        eps=eps_config,
        d_model=sb_vocab_size,
        n_warmup_steps=warmup_steps_config, 
        lr_factor=lr_factor_config
    )

    # initialize criterion (loss function)
    criterion = nn.CrossEntropyLoss(ignore_index=sb_vocab_dict['<mask>'])

    # initialize beam search values
    beam_size = beam_size_config
    len_penalty_alpha = len_penalty_alpha_config
    max_len_a = max_len_a_config
    max_len_b = max_len_b_config

    # START TRAINING

    # start training
    batch_start = 0
    while True:
        print(f"Inside while with batch_start = {batch_start}")
        try:
            print("Starting training!")
            train_transformer(model, train_dataloader, val_dataloader, num_epochs,
                             save_path, save_interval, optimizer, criterion, sp, es_patience=5, avg_n_weights=5,
                             device='cuda')
        except torch.cuda.OutOfMemoryError as e:
            print(f"Skipping to: {batch_start}")
            torch.cuda.empty_cache()
            continue