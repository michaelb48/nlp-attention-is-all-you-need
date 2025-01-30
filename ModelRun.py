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
from Optimizer import CustomOptim
from itertools import islice
import json
import csv
from torchtext.data.metrics import bleu_score
from utils import set_seed, ensure_directory_exists, save_checkpoint

# this is the path to the experiment configuration; set the values in the config file to execute a new experiment
CONFIG_FILE = "ex_config-1-extension"
CONFIG_PATH = "config"

def train_fn(config_file, model, dataloader, optimizer, criterion, device, clip, save_path_prefix, save_interval_in_minutes,total_training_steps,results,epoch,max_train_loop_steps):
    global in_eval
    if in_eval:
        return
    model.train()
    total_loss = 0
    tk0 = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    output = None

    global batch_start

    last_save_time = time.time()
    
    # caculate how many batches are left in this epoch
    step = batch_start % max_train_loop_steps
    for batch_idx, batch in enumerate(islice(tk0, step, max_train_loop_steps)):

        # in case the loop gets restarted we have to prematurely stop the training loop
        if step >= max_train_loop_steps:
            break

        # in case we reach the end point before the entire epoch training is completed
        if batch_start >= total_training_steps:
            break
        
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
        batch_start += 1
        step += 1

        output = output.argmax(dim=-1)
        
        # backward pass
        loss.backward()
        # clip gradients to avoid exploding gradients issue
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # update model parameters
        optimizer.step()

        # Save progress for plotting
        if step % 100 == 0:
            print(
                f'Batch: {step+1}, Loss: {loss.item():.4f}, Learning Rate: {optimizer.get_lr():.7f}')
            # open writer to training results
            with open(results, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, step, loss.item(), optimizer.get_lr()])
            
        # Save model every interval
        if time.time() - last_save_time >= save_interval_in_minutes * 60:
            save_checkpoint(model,optimizer,epoch,save_path_prefix,step,config_file)
            last_save_time = time.time()

        tk0.set_postfix(loss=total_loss / max_train_loop_steps)
    tk0.close()
    perplexity = np.exp(total_loss / step)

    return perplexity

def eval_fn(config_file, model, dataloader, criterion, device, sp, epoch,max_train_loop_steps,
                                              beam_size, len_penalty_alpha, max_len_a, max_len_b):
    model.eval()
    total_loss = 0.0
    steps = 0
    hypotheses = []
    references = []
    global in_eval

    in_eval = True
    tk0 = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    total_steps = int(max_train_loop_steps*0.3)
    with torch.no_grad():
        for batch in islice(tk0, 0, total_steps):
            source = batch[0].to(device)
            target = batch[1].to(device)

            # forward pass
            optimizer.zero_grad()
            output = model(source, target[:, :-1])
            #translation = model.translate(source,beam_size, len_penalty_alpha, max_len_a, max_len_b)

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
            target_tokens = sp.encode_as_pieces(sp.decode(target[0].cpu().tolist()))
            translation_tokens = sp.encode_as_pieces(sp.decode(output[0].cpu().tolist()))
            
            print("Expected Output:", target_tokens)
            print("Predicted Output:", translation_tokens)
            
            hypotheses += translation_tokens
            references += [[token] for token in target_tokens if token != '<mask>']
            
            tk0.set_postfix(loss=total_loss / steps)
    tk0.close()
    perplexity = np.exp(total_loss / total_steps)
    references = [[[item[0] for item in references]]]
    hypotheses = [hypotheses]
    # Compute the BLEU score
    bleu = bleu_score(candidate_corpus=hypotheses, references_corpus=references)

    in_eval = False
    
    return perplexity, bleu


def train_transformer(config_file, model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs, total_training_steps,
                      save_path_prefix, save_interval_in_minutes, results_save_path, average_model_weight_num, sp, device,
                     beam_size, len_penalty_alpha, max_len_a, max_len_b,es_patience=5):
    
    global best_bleu
    patience = 0
    clip = 1.0
    max_train_loop_steps = total_training_steps // num_epochs
    global epoch_start
    
    for epoch in range(epoch_start, num_epochs+1):

        # one epoch training
        train_perplexity = train_fn(config_file, model, train_dataloader, optimizer, criterion, device, clip, save_path_prefix, save_interval_in_minutes,total_training_steps,os.path.join(results_save_path,f"{config_file}_train_results.csv"),epoch,max_train_loop_steps)
        
        # one epoch validation
        valid_perplexity, valid_bleu = eval_fn(config_file, model, val_dataloader, criterion, device, sp, epoch,max_train_loop_steps,
                                              beam_size, len_penalty_alpha, max_len_a, max_len_b)
        
        print(f'Epoch: {epoch}, Train perplexity: {train_perplexity:.4f}, Valid perplexity: {valid_perplexity:.4f}, Valid BLEU4: {valid_bleu:.4f}')
        with open(os.path.join(results_save_path, f"{config_file}_validation_results.csv"), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_perplexity,valid_perplexity, valid_bleu])
        
        # early stopping mechanism
        is_best = valid_bleu > best_bleu
        if is_best:
            print(f'BLEU score improved ({best_bleu:.4f} -> {valid_bleu:.4f}). Saving Model!')
            best_bleu = valid_bleu
            patience = 0
            save_checkpoint(model, optimizer, epoch, save_path_prefix, config_file=config_file)
        else:
            patience += 1
            print(f'Early stopping counter: {patience} out of {es_patience}')
            if patience == es_patience:
                print(f'Early stopping! Best BLEU: {best_bleu:.4f}')
                break
        epoch_start +=1
    return model


if __name__ == '__main__':
    # set random seed for reproducability
    set_seed(2630)
    
    # Open and load the JSON file into a dictionary
    config_path = os.path.join(CONFIG_PATH,f"{CONFIG_FILE}.json")
    with open(config_path, 'r') as file:
        config = json.load(file)

    # VARIABLES FROM CONFIG FILE THAT CONTROL EXPERIMENT RUN
    pytorch_cuda_config = config.get('pytorch_cuda','max_split_size_mb:128')
    
    corpus_path_config = config.get('corpus_path','/corpus/df_encoded.pkl')
    bpe_model_path_config = config.get('bpe_model_path','/bpe/bpe_model.model')
    results_path_config = config.get('results','results')
    
    batch_size_config = config.get('batch_size',16)
    dataset_value_split_config = config.get('dataset_value_split',0.1)

    lr_config = config.get('lr',1e-4)
    beta1_config = config.get('beta1',0.9)
    beta2_config = config.get('beta2',0.98)
    eps_config = config.get('eps',1e-9)
    warmup_steps_config = config.get('warmup_steps',4000)
    lr_factor_config = config.get('lr_factor',1)

    num_epochs_config = config.get('num_epochs', 10)
    total_training_steps_config = config.get('total_training_steps', 100000)
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
    if t_dot_product_config == 1:
        t_dot_product_config = True
    else:
        t_dot_product_config = False
    label_smoothing_config = config.get('label_smoothing',0.1)

    beam_size_config = config.get('beam_size',4)
    len_penalty_alpha_config = config.get('len_penalty_alpha','max_split_size_mb:128')
    max_len_a_config = config.get('max_len_a','max_split_size_mb:128')
    max_len_b_config = config.get('max_len_b','max_split_size_mb:128')
    
        
    # INIT EXPERIMENT RUN
    
    # set general training parameters
    num_epochs = num_epochs_config
    total_training_steps = total_training_steps_config
    model_save_path = model_save_path_config
    results_save_path = results_path_config
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
    
    # initialize the optimizer
    print("Initializing optimizer ...")
    optimizer = CustomOptim(
        optimizer=torch.optim.Adam(model.parameters(), lr=lr_config, betas=(beta1_config, beta2_config), eps=eps_config),
        lr=lr_config,
        beta1=beta1_config,
        beta2=beta2_config,
        eps=eps_config,
        d_model=sb_vocab_size,
        n_warmup_steps=warmup_steps_config, 
        lr_factor=lr_factor_config
    )

    # initialize criterion (loss function)
    criterion = nn.CrossEntropyLoss(ignore_index=sb_vocab_dict['<mask>'],label_smoothing=label_smoothing_config)

    # initialize beam search values
    beam_size = beam_size_config
    len_penalty_alpha = len_penalty_alpha_config
    max_len_a = max_len_a_config
    max_len_b = max_len_b_config

    # make sure the directories for storing the models and results exist
    ensure_directory_exists(results_save_path)
    ensure_directory_exists(model_save_path)
                            
    # create results file for training and validation to ease plotting
    train_results_path = os.path.join(results_save_path, f"{CONFIG_FILE}_train_results.csv")
    validation_results_path = os.path.join(results_save_path, f"{CONFIG_FILE}_validation_results.csv")

    # create the files with headers
    with open(train_results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Step", "Loss", "Learning Rate"])
    with open(validation_results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch","TrainPerplexity", "ValidPerplexity", "Bleu"])

    
    # START TRAINING
    # Training tracking
    batch_start = 0
    epoch_start = 0
    in_eval = False
    best_bleu = float('-inf')
    start_time = time.time()  # Total training start time
    
    while True:
        print(f"Inside while with batch_start = {batch_start}")
        try:
            print("Starting training!")
            train_transformer(config_file=CONFIG_FILE,
                               model=model,
                               optimizer=optimizer,
                               criterion=criterion,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               num_epochs=num_epochs,
                               total_training_steps=total_training_steps,
                               save_path_prefix=model_save_path,
                               save_interval_in_minutes=save_interval_in_minutes,
                               results_save_path=results_save_path,
                               average_model_weight_num=average_model_weight_num,
                               sp=sp,
                               es_patience=5,
                               device=device,
                               beam_size=beam_size,
                               len_penalty_alpha=len_penalty_alpha,
                               max_len_a=max_len_a,
                               max_len_b=max_len_b
                              )
        except torch.cuda.OutOfMemoryError as e:
            print(f"Skipping to: {batch_start}")
            torch.cuda.empty_cache()
            continue

        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print(f"The complete training took {hours:02}:{minutes:02}:{seconds:02} (HH:MM:SS).")
        save_checkpoint(model, optimizer, 'end', model_save_path, config_file=CONFIG_FILE)
        break