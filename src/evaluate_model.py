import pdb

import torch
from torch.utils.data import DataLoader
import pickle
import logging
import numpy as np
import json
from transformer import Transformer
import os
from transformer_dataset import TransformerDataset
from datetime import datetime


class EvaluateModel:

    def __init__(self):
        self.training_val_loss = float('inf')
        self.device = "cpu"
        #self.base_path = "/home/pv10123z/mldm-project/"
        self.base_path = "/Users/victor.penaloza/Documents/mldm-project/"

        self.model_path = os.path.join(self.base_path,"out/12-12-2024_20-50-18")

        self.data_dir = "data/train"
        self.log_dir = "logs"

        self.start_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.log_path = os.path.join(self.base_path, self.log_dir, self.start_timestamp + "_evaluation.log")

        self.historic_path = os.path.join(self.base_path, self.data_dir, "historic_T3_V384_P1.pkl")
        self.next_path = os.path.join(self.base_path, self.data_dir, "next_V384_P1.pkl")

        logging.basicConfig(filename=self.log_path, filemode='a', level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
        self.logger = logging.getLogger(__name__)

        self.batch_size = 64

    def get_data_from_pkl(self):
        # Load historic: X
        with open(self.historic_path, 'rb') as f:
            input_sequences = pickle.load(f)

        # Load next future transactions: Y
        with open(self.next_path, 'rb') as f:
            target_embeddings = pickle.load(f)

        self.logger.info(f"Reading test data from .pkl : SUCCESS ")

        return input_sequences, target_embeddings

    def load_data(self):
        input_sequences, target_embeddings = self.get_data_from_pkl()
        # A 3 dimension vector is required for the labels (samples, 1, embedding_size)
        target_embeddings = np.expand_dims(target_embeddings, axis=1)

        # Create Datasets
        test_dataset = TransformerDataset(input_sequences, target_embeddings)

        # Create Dataloaders
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.logger.info(f"Creating test dataloader : SUCCESS ")

        return test_dataloader

    def load_model(self):
        with open(os.path.join(self.model_path, "model_config.json"), 'r') as f:
            self.model_config = json.load(f)

        model = Transformer(self.model_config).double().to(self.device)
        checkpoint = torch.load(os.path.join(self.model_path, "checkpoint.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])

        self.training_val_loss = checkpoint["val_loss"]

        self.logger.info(f"Pretrained model {self.model_path} loaded : SUCCESS ")

        return model

    def evaluate(self, test_dataloader, model):
        criterion = torch.nn.MSELoss()
        similarity_criterion = torch.nn.CosineSimilarity(dim=-1)

        test_loss = 0
        similarity_loss = 0

        predicted_embeddings = list()

        # Evaluation Loop
        model.eval()
        with torch.no_grad():
            for _input, target in test_dataloader:
                src = _input.double().to(self.device)
                target = target.double().to(self.device)

                prediction = model(src, self.device)
                loss = criterion(prediction, target[:, :, :])
                test_loss += loss.item()

                # Compute Similarity Loss
                cosine_similarity = similarity_criterion(prediction, target)
                cosine_difference = 1 - cosine_similarity
                mean_cosine_difference = cosine_difference.mean()
                similarity_loss += mean_cosine_difference.item()  # Accumulate the average similarity per batch

                # Store predictions
                predicted_embeddings.append(prediction)

        test_loss /= len(test_dataloader)
        similarity_loss /= len(test_dataloader)

        predicted_embeddings = torch.cat(predicted_embeddings, dim=0)

        return test_loss, similarity_loss, predicted_embeddings


def predict_next_purchases(self, predicted_embeddings, customer_ids):

    recommendations = []
    for customer_id, embedding in zip(customer_ids, predicted_embeddings):
        # Anna's search function that receives the predicted embedding and finds the corresponding top 10 products
        top_products = search_function(embedding)
        for rank, product_id in enumerate(top_products, start=1):
            recommendations.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'rank': rank
            })

    # Output a DataFrame from the recommendations list
    recommendations_df = pd.DataFrame(recommendations)

    return recommendations



def main():
    evaluator = EvaluateModel()

    model = evaluator.load_model()
    test_dataloader = evaluator.load_data()

    mse_loss, cosine_loss, predicted_embeddings = evaluator.evaluate(test_dataloader, model)

    evaluator.logger.info(f"Average MSE Loss: {mse_loss}, Average Cosine Similarity Loss: {cosine_loss}, Training Loss: {evaluator.training_val_loss}")


if __name__ == "__main__":
    main()
