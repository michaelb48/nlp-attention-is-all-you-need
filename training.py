import torch.nn as nn
import torch
import tqdm

def train_fn(model, dataloader, optimizer, scheduler, criterion, clip=1.0):
    model.train()
    total_loss = 0
    steps = 0
    tk0 = tqdm(dataloader, total=len(dataloader), position=0, leave=True)

    for batch in tk0:
        #print(batch[0].shape)
        #print(batch[1].shape)
        #source, source_lengths = batch.src
        #target, target_lengths = batch.trg
        source = batch[0].to(device)
        target = batch[1].to(device)
        #print(f"source: {source}")
        #print(f"target: {target}")
        #print(f"labels: {labels}")
        #print(f"labels: {labels.shape}")
        # source: (batch_size, source_seq_len), source_lengths: (batch_size)
        # target: (batch_size, target_seq_len), target_lengths: (batch_size)
        
        # forward pass
        #print(f"target[:, :-1]: {target[:, :-1].shape}")
        #print(f"target[:, 1:]: {target[:, 1:].shape}")
        #print(f"target[:, :-1]: {target[:, :-1]}")
        #print(f"target[:, 1:]: {target[:, 1:]}")
        optimizer.zero_grad()
        output = model(source, target[:, :-1])  # (batch*size, target_seq_len - 1, vocab_size)
        
        # calculate the loss
        loss = criterion(
            output.view(-1, output.size(-1)),  # (batch_size * (target_seq_len - 1), vocab_size)
            target[:, 1:].contiguous().view(-1)  # (batch_size * (target_seq_len - 1))
        )
        total_loss += loss.item()
        steps += 1
        output = output.argmax(dim=-1)  # (batch_size, target_seq_len - 1)
        # backward pass
        loss.backward()
        # clip gradients to avoid exploding gradients issue
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        # update model parameters
        optimizer.step()
        scheduler.step()

        if steps % 100 == 0:
            print(f'Step: {steps}, Loss: {loss.item():.4f}, '
                f'Learning Rate: {scheduler.get_lr()[0]:.7f}')
        
        tk0.set_postfix(loss=total_loss/steps)
    tk0.close()
    perplexity = np.exp(total_loss / len(dataloader))
    
    return output, perplexity

def eval_fn(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    steps = 0
    hypotheses = []
    references = []
    tk0 = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    with torch.no_grad():
        for batch in tk0:
            #source, source_lengths = batch.src
            #target, target_lengths = batch.trg
            source = batch[0].to(device)
            target = batch[1].to(device)
            # source: (batch_size, source_seq_len), source_lengths: (batch_size)
            # target: (batch_size, target_seq_len), target_lengths: (batch_size)
            
            # forward pass
            optimizer.zero_grad()
            output = model(source, target[:, :-1])  # (batch*size, target_seq_len - 1, vocab_size)
            
            # calculate the loss
            loss = criterion(
                output.view(-1, output.size(-1)),  # (batch_size * (target_seq_len - 1), vocab_size)
                target[:, 1:].contiguous().view(-1)  # (batch_size * (target_seq_len - 1))
            )
            total_loss += loss.item()
            steps += 1
            output = output.argmax(dim=-1)  # (batch_size, target_seq_len - 1)
            print(f"output: {output}")
            target = target[:, 1:]  # (batch_size, target_seq_len - 1)
            # converting the ids to tokens (used later for calculating BLEU score)
            #pred_tokens = convert_ids_to_text(output, de_text.vocab, EOS_IDX, UNK_IDX)
            #target_tokens = convert_ids_to_text(target, de_text.vocab, EOS_IDX, UNK_IDX)
            pred_tokens = sp.encode_as_pieces(sp.decode(output[0].cpu().tolist()))
            target_tokens = sp.encode_as_pieces(sp.decode(target[0].cpu().tolist()))
            print("Expected Output:", target_tokens)
            print("Predicted Output:", pred_tokens)
            hypotheses += pred_tokens
            references += [[token] for token in target_tokens if token != '<mask>']
            tk0.set_postfix(loss=total_loss/steps)
    tk0.close()
    perplexity = np.exp(total_loss / len(dataloader))
    references = [[[item[0] for item in references]]]
    hypotheses = [hypotheses]
    #print(f"hypotheses: {hypotheses}")
    #print(f"references: {references}")
    bleu4 = bleu_score(hypotheses, references)
    
    return output, perplexity, bleu4