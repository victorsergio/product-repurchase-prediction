import torch
import numpy as np
from torch.utils.data import Dataset



class RankingModelDataset(Dataset):
    def __init__(self, X1, X2, y, products_to_embeddings_dict, unique_sequences_id_to_embedding_dict):

        self.X1 = X1
        self.X2 = X2
        self.y = torch.tensor(y, dtype=torch.float32)
        self.products_to_embeddings_dict = products_to_embeddings_dict
        self.unique_sequences_id_to_embedding_dict = unique_sequences_id_to_embedding_dict

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X1 = self.unique_sequences_id_to_embedding_dict[self.X1[idx]]
        X2 = self.products_to_embeddings_dict[self.X2[idx]]

        return torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32), self.y[idx]