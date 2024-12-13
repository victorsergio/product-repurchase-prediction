import torch
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    def __init__(self, input_sequences, target_embeddings):
        """
        Args:
            input_sequences (numpy array): 3D array of input sequences (num_customers, timesteps, embedding_size)
            target_embeddings (numpy array): 2D array of target embeddings (num_customers, embedding_size)
        """
        self.input_sequences = torch.tensor(input_sequences, dtype=torch.float32)
        self.target_embeddings = torch.tensor(target_embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_embeddings[idx]
