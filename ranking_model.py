import torch
import torch.nn as nn


class RankingModel(nn.Module):
    def __init__(self, model_config):
        super(RankingModel, self).__init__()
        embedding_dim = model_config["embedding_dim"]
        #hidden_dim = model_config["hidden_dim"]
        fc_dim = model_config["fc_dim"]


        #self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(embedding_dim * 2, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, transactions, product_embedding):
        # Process transaction embeddings with LSTM
        #_, (hidden_state, _) = self.lstm(transactions)  # hidden_state: [1, batch, hidden_dim]
        #hidden_state = hidden_state.squeeze(0)  # [batch, hidden_dim]
        # Compute the average embedding across the timesteps
        avg_transaction_embedding = transactions.mean(dim=1)  # [batch, embedding_dim]

        # Concatenate hidden state with product embedding
        x = torch.cat((avg_transaction_embedding, product_embedding), dim=1)  # [batch, hidden_dim + embedding_dim]

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # Output probability
        return self.sigmoid(x)
