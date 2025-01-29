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
from utils import set_seed, ensure_directory_exists, save_checkpoint, load_checkpoint

# this is the path to the test configuration; set the values in the config file to execute a new test
CONFIG_FILE = "ex_config-1"
CONFIG_PATH = "config"
TEST_MODEL_PATH = "~/groups/192.039-2024W/attentiondeficit/test-results/models/ex_config-1_avg_weights_model"


def test_fn(model,device,criterion,beam_size,len_penalty_alpha,max_len_a,max_len_b,sp,total_test_steps):
                                   
    model.eval()
    total_loss = 0.0
    steps = 0
    hypotheses = []
    references = []
    
    tk0 = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    
    with torch.no_grad():
        for batch in islice(tk0, 0, total_test_steps):

            # move sequences to device
            source = batch[0].to(device)
            target = batch[1].to(device)

            # forward pass
            optimizer.zero_grad()
            output = model(source, target[:, :-1])
            translation = model.beam_search_translate(source, beam_size, len_penalty_alpha, max_len_a, max_len_b)

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
            target_tokens = sp.encode_as_pieces(sp.decode(target.cpu().tolist()))
            pred_tokens = sp.encode_as_pieces(sp.decode(output.cpu().tolist()))
            translation_tokens = sp.encode_as_pieces(sp.decode(translation.cpu().tolist()))
            
            hypotheses += translation_tokens
            references += [[token] for token in target_tokens if token != '<mask>']
            
            tk0.set_postfix(loss=total_loss / steps)
    tk0.close()
    perplexity = np.exp(total_loss / total_test_steps)
    references = [[[item[0] for item in references]]]
    hypotheses = [hypotheses]
    # Compute the BLEU score
    bleu = bleu_score(candidate_corpus=hypotheses, references_corpus=references)
    
    return perplexity, bleu


def test_transformer(model, device, criterion, test_dataloader, beam_size, len_penalty_alpha, max_len_a, max_len_b, sp, total_test_steps, test_results_path):

    # Testing tracking
    attempt = 1
    bleu = float('-inf')
    perplexity = float('-inf')
    testing_start_time = time.time()  # Total training start time
    
    while True:
        try:
            print("Starting testing!")
            attempt_start_time = time.time()
            perplexity, bleu = test_fn(model=model,
                                        device=device,
                                        dataloader=test_dataloader,
                                        criterion=criterion,
                                        beam_size = beam_size,
                                        len_penalty_alpha = len_penalty_alpha,
                                        max_len_a = max_len_a,
                                        max_len_b = max_len_b,
                                        sp=sp,
                                        total_test_steps = total_test_steps)

            elapsed_time = time.time() - attempt_start_time
            
            print(f'Time in sec: {elapsed_time}, Test perplexity: {perplexity:.4f}, Test BLEU: {bleu:.4f}')
            with open(test_results_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([elapsed_time, perplexity, bleu])
            return elapsed_time, perplexity, bleu
        except torch.cuda.OutOfMemoryError as e:
            if attempt == 5:
                print(f"All test attempts failed due to memory issues. No results available.")
                return time.time() - testing_start_time, perplexity, bleu
            print(f"Test attempt {attempt} failed due to memory issues. {5-attempt} attempts remaining.")
            torch.cuda.empty_cache()
            attempt += 1
            continue

if __name__ == '__main__':
    # set random seed for reproducability
    set_seed(2630)
    
    # Open and load the JSON file into a dictionary
    config_path = os.path.join(CONFIG_PATH,f"{CONFIG_FILE}.json")
    with open(config_path, 'r') as file:
        config = json.load(file)

    # VARIABLES FROM CONFIG FILE THAT CONTROL EXPERIMENT RUN
    pytorch_cuda_config = config.get('pytorch_cuda','max_split_size_mb:128')

    corpus_path_config = config.get('testcorpus_path','/test_corpus/df_encoded.pkl')
    bpe_model_path_config = config.get('bpe_model_path','/bpe/bpe_model.model')
    results_path_config = config.get('results_path','results')
    
    batch_size_config = config.get('batch_size',16)
    dataset_value_split_config = config.get('dataset_value_split',0.3)

    label_smoothing_config = config.get('label_smoothing',0.1)
    
    total_test_steps_config = config.get('total_test_steps', 30000)

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

    # INIT TEST RUN
    
    # set general training parameters
    total_test_steps = total_test_steps_config
    results_save_path = results_path_config
    
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
    test_dataloader, _ = create_train_val_dataloaders(
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

    # loading the model
    print("Loading model ...")
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

    model = load_checkpoint(file_path=TEST_MODEL_PATH, model=model, device=device).to(device)

    # initialize criterion (loss function)
    criterion = nn.CrossEntropyLoss(ignore_index=sb_vocab_dict['<mask>'],label_smoothing=label_smoothing_config)

    # initialize beam search values
    beam_size = beam_size_config
    len_penalty_alpha = len_penalty_alpha_config
    max_len_a = max_len_a_config
    max_len_b = max_len_b_config

    # make sure the directories for storing the results exist
    ensure_directory_exists(results_save_path)
                            
    # create results file for testing to ease plotting
    test_results_path = os.path.join(results_save_path, f"{CONFIG_FILE}_test_results.csv")

    # create the files with headers
    with open(test_results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time","perplexity", "bleu"])
    
    # START TESTING
    elapsed_time, perp, bleu = test_transformer(model=model,
                                                device=device,
                                                test_dataloader=test_dataloader,
                                                criterion=criterion,
                                                beam_size = beam_size,
                                                len_penalty_alpha = len_penalty_alpha,
                                                max_len_a = max_len_a,
                                                max_len_b = max_len_b,
                                                sp=sp,
                                                total_test_steps = total_test_steps_config,
                                                test_results_path = test_results_path)
    
    # Calculate elapsed time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"The complete test took {hours:02}:{minutes:02}:{seconds:02} (HH:MM:SS).")
    print(f'The model achieved testing perplexity: {perp:.4f} and testing BLEU: {bleu:.4f}')
