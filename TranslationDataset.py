import torch
from torch.utils.data import Dataset, DataLoader, random_split

class TranslationDataset(Dataset):
    def __init__(self, dataframe, vocab, start_token="<s>", end_token="</s>", pad_token="<mask>"):
        self.en_sentences = dataframe['en'].tolist()
        self.de_sentences = dataframe['de'].tolist()
        # filter all sequences that could cause memory issues
        index_to_remove = set()
        for i in range(len(self.en_sentences)):
            if len(self.en_sentences[i]) > 4998 or len(self.de_sentences[i]) > 4998:
                index_to_remove.add(i)
        # create new lists without the long sequences
        self.en_sentences = [seq for idx, seq in enumerate(self.en_sentences) if idx not in index_to_remove]
        self.de_sentences = [seq for idx, seq in enumerate(self.de_sentences) if idx not in index_to_remove]
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        # Adding start and end tokens
        en_tokens = [self.start_token] + self.en_sentences[idx] + [self.end_token] # sos + sentence + eos
        de_tokens = [self.start_token] + self.de_sentences[idx] + [self.end_token] # sos + sentence + eos

        return {
            'en': en_tokens,
            'de': de_tokens
        }

def collate_batch(batch, vocab):
    en_sequences = []
    de_sequences = []

    # Replace piece tokens with IDs and for unknown pieces, use the <unk> ID
    for item in batch:
        en_indices = torch.tensor([vocab.get(token, vocab['<unk>']) for token in item['en']])
        de_indices = torch.tensor([vocab.get(token, vocab['<unk>']) for token in item['de']])

        en_sequences.append(en_indices)
        de_sequences.append(de_indices)

    # Find the maximum length in either language
    max_len_en = max(len(seq) for seq in en_sequences)
    max_len_de = max(len(seq) for seq in de_sequences)
    max_len = max(max_len_en, max_len_de)

    # Pad sequences based on the maximum length
    en_padded = torch.stack([torch.cat([seq, torch.full((max_len_en - len(seq),), vocab['<mask>'])]) for seq in en_sequences])
    de_padded = torch.stack([torch.cat([seq, torch.full((max_len_de - len(seq),), vocab['<mask>'])]) for seq in de_sequences])

    # Return padded tensors and original sequence lengths
    return en_padded, de_padded, torch.tensor([len(seq) for seq in en_sequences]), torch.tensor([len(seq) for seq in de_sequences])

def create_train_val_dataloaders(dataset, batch_size, vocab, val_split=0.1, shuffle=True):
    val_length = int(len(dataset) * 0)
    train_length = len(dataset) - val_length
    
    dataset, _ = random_split(
        dataset, 
        [train_length, val_length],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    val_length = int(len(dataset) * val_split)
    train_length = len(dataset) - val_length

    print(f"train length: {train_length}, val length: {val_length}")
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_length, val_length],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_batch(b, vocab)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=lambda b: collate_batch(b, vocab)
    )
    return train_dataloader, val_dataloader