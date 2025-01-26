import deepspeed
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Assuming you have your Transformer model implemented as `Transformer`
model = Transformer(
    n_vocab_len=37000,
    i_vocab_padding=0,
    d_model=512,
    device="cuda"
).to("cuda")

# Custom Dataset (Placeholder)
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Dummy data (replace with real data)
train_data = [
    (torch.randint(0, 37000, (50,)), torch.randint(0, 37000, (50,))) for _ in range(1000)
]
train_dataset = CustomDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Loss function
criterion = CrossEntropyLoss(ignore_index=0)

# DeepSpeed Configuration
ds_config = "ds_config.json"  # Path to your DeepSpeed config file
parameters = filter(lambda p: p.requires_grad, model.parameters())

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=parameters,
    config=ds_config
)

# Training Loop
epochs = 3
for epoch in range(epochs):
    model_engine.train()
    for step, (src, tgt) in enumerate(train_dataloader):
        src, tgt = src.to("cuda"), tgt.to("cuda")
        output = model_engine(src, tgt)

        # Flatten the logits and labels for CrossEntropy
        logits = output.view(-1, output.size(-1))
        labels = tgt.view(-1)

        # Compute loss
        loss = criterion(logits, labels)

        # Backpropagation and optimization
        model_engine.backward(loss)
        model_engine.step()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")



# From Janick's notebook

# Loading BPE model
sp = spm.SentencePieceProcessor()
sp.load('bpe_model.model')

sp = spm.SentencePieceProcessor()
sp.load('bpe_model.model')
sb_vocab_size = sp.get_piece_size()
sb_vocab = [sp.id_to_piece(i) for i in range(sb_vocab_size)]
sb_vocab_dict = {sb_vocab[i]: i for i in range(sb_vocab_size)}

# Loading dataset
df_encoded = pd.read_pickle("df_encoded.pkl")
dataset = TranslationDataset(df_encoded, sb_vocab)
train_dataloader, valid_dataloader = create_train_val_dataloaders(
        dataset,
        batch_size=32,
        vocab=sb_vocab_dict,
        val_split=0.1
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = Transformer(n_vocab_len=sb_vocab_size,i_vocab_padding = sb_vocab_dict['<mask>']).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=5)  # Ignore padding index
optimizer = torch.optim.Adam(model.parameters(), lr=0.044, betas=(0.9, 0.98), eps=1e-9)
    
# Learning rate scheduler
# def lr_lambda(step):
#     warmup_steps = 4000
#     step = max(1, step)
#     return min(step ** (-0.5), step * warmup_steps ** (-1.5))

# scheduler = LambdaLR(optimizer, lr_lambda)

best_bleu4 = float('-inf')
es_patience = 3
patience = 0
model_path = 'model.pth'
N_EPOCHS = 10
CLIP = 1.0

for epoch in range(0, N_EPOCHS + 1):
    # one epoch training
    _, train_perplexity = train_fn(model, train_dataloader, optimizer, criterion, CLIP)
    
    # one epoch validation
    _, valid_perplexity, valid_bleu4 = eval_fn(model, valid_dataloader, criterion)
    
    print(f'Epoch: {epoch}, Train perplexity: {train_perplexity:.4f}, Valid perplexity: {valid_perplexity:.4f}, Valid BLEU4: {valid_bleu4:.4f}')
    
    # early stopping
    is_best = valid_bleu4 > best_bleu4
    if is_best:
        print(f'BLEU score improved ({best_bleu4:.4f} -> {valid_bleu4:.4f}). Saving Model!')
        best_bleu4 = valid_bleu4
        patience = 0
        torch.save(model.state_dict(), model_path)
    else:
        patience += 1
        print(f'Early stopping counter: {patience} out of {es_patience}')
        if patience == es_patience:
            print(f'Early stopping! Best BLEU4: {best_bleu4:.4f}')
            break