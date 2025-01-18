import pdb
import time
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
import pandas as pd
from qdrant_api import QdrantAPI
import argparse
from utils import setup_logger
from sklearn.metrics.pairwise import cosine_similarity

class EvaluateModel:

    def __init__(self):
        self.training_val_loss = float('inf')
        self.device = "cpu"
        self.base_path = "/Users/Documents/mldm-project/"
        #self.base_path = "/home/pv10123z/mldm-project/"
        #self.base_path = "/content/drive/MyDrive/MLDM Project/SLURM/"
        self.model_path = os.path.join(self.base_path,"out","06-01-2025_01-50-07")

        self.data_dir = "data/test"
        self.log_dir = "logs"

        self.start_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.log_path = os.path.join(self.base_path, self.log_dir, self.start_timestamp + "_evaluation.log")

        self.evaluation_dir = "evaluation_out"
        self.evaluation_path = os.path.join(self.base_path, self.evaluation_dir,
                                            self.start_timestamp)
        self.similar_vectors_file = os.path.join(self.evaluation_path,"similar_vectors.pkl")
        # Create a new directory for this evaluation run
        os.makedirs(self.evaluation_path, exist_ok=True)

        #self.data_name = "T10_V384_P1.pkl"
        self.historic_path = os.path.join(self.base_path, self.data_dir, f"X1_test.pkl")
        self.next_path = os.path.join(self.base_path, self.data_dir, f"y_test.pkl")

        #self.historic_path = os.path.join(self.base_path, "data/train", "X_V200_P1.pkl")
        #self.next_path = os.path.join(self.base_path, "data/train", "y_V200_P1.pkl")


        self.customers_path = os.path.join(self.base_path, self.data_dir, f"customers_ids_test.pkl")
        self.products_lists_path = os.path.join(self.base_path, self.data_dir, f"products_ids_test.pkl")

        self.logger = setup_logger(self.log_path, to_console=True, logger_name=__name__)

        self.batch_size = 64

        self.fasttext_embedding_size = 200

    # def build_test_data(self):
    #     df = pd.read_parquet(os.path.join(self.base_path, "data/train", "test_historic.parquet"))
    #     pdb.set_trace()
    #
    #     customers_ids = df.customer_id
    #     target_embeddings = df.transaction_embedding
    #     products_lists = df.products
    #     transactions_ids = df.transaction_id
    #     dates = df.date
    #
    #     return customers_ids, products_lists, target_embeddings, transactions_ids, dates

    def get_data_from_pkl(self):
        # Load historic: X
        with open(self.historic_path, 'rb') as f:
            input_sequences = pickle.load(f)

        # Load next future transactions: Y
        with open(self.next_path, 'rb') as f:
            target_embeddings = pickle.load(f)

        # Load customers ids
        with open(self.customers_path, 'rb') as f:
            customers = pickle.load(f)

        # Load products list
        with open(self.products_lists_path, 'rb') as f:
            products_lists = pickle.load(f)

        self.logger.info(f"Reading test data from .pkl : SUCCESS ")

        return input_sequences, target_embeddings, customers, products_lists

    def load_data(self):
        input_sequences, target_embeddings, customers, products_lists = self.get_data_from_pkl()
        #input_sequences, target_embeddings = self.get_data_from_pkl()

        # A 3 dimension vector is required for the labels (samples, 1, embedding_size)
        target_embeddings = np.expand_dims(target_embeddings, axis=1)

        # Create Datasets
        test_dataset = TransformerDataset(input_sequences, target_embeddings)

        # Create Dataloaders
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.logger.info(f"Creating test dataloader : SUCCESS ")

        return test_dataloader, customers, products_lists, target_embeddings
        #return test_dataloader, None, None, target_embeddings

    def load_model(self):
        with open(os.path.join(self.model_path, "model_config.json"), 'r') as f:
            self.model_config = json.load(f)

        model = Transformer(self.model_config).double().to(self.device)
        checkpoint = torch.load(os.path.join(self.model_path, "checkpoint.pth"), map_location=torch.device(self.device))
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


    def get_similar_embeddings_to_disk(self, predicted_embeddings, filter=None):
        qdrant_api = QdrantAPI(
                base_path=self.base_path,
                #data_name=self.historic_filename + ".parquet",
                collection_name="transactions_embeddings_fasttext",
                vector_size=self.fasttext_embedding_size
            )

        # Perform similarity search
        similar_vectors = qdrant_api.perform_similarity_search(predicted_embeddings, top_k=10, filter=None, batch_size=500)

        with open(self.similar_vectors_file, 'wb') as f:
            pickle.dump(similar_vectors, f)

        self.logger.info(
            f"Similar queried vectors saved to disk: {self.similar_vectors_file}, SUCCESS")

        return similar_vectors

    def get_similar_embeddings_from_disk(self, file_dir):
        similar_vectors_file = os.path.join(self.base_path, self.evaluation_dir,file_dir,"similar_vectors.pkl")

        # Load precalculated similar vectors
        with open(similar_vectors_file, 'rb') as f:  # 'rb' means "read in binary mode"
            similar_vectors = pickle.load(f)

        return similar_vectors

    @staticmethod
    def loop_over_response(response, calculation_method):
        """
            Traverse the response by element, extract the top 10 product lists for each id,
            and pass them to the calculation_method.

            Parameters:
                response: List of QueryResponse objects
                calculation_method: Function that receives a list of lists of products
            """
        results = list()
        # Flatten the Qadrant original response, because it cames in batches.
        response = [element for batch in response for element in batch]
        #pdb.set_trace()

        for query_response in response:
            product_lists = []
            for scored_point in query_response.points:
                products = scored_point.payload['products_list'].split(", ")
                product_lists.append(products)

            results.append( calculation_method(product_lists))
            #pdb.set_trace()

        #pdb.set_trace()
        return results


    @staticmethod
    def sequential_search(product_lists, total_limit=100):
        #pdb.set_trace()
        selected_products = []  # List to keep track of selected unique products
        seen_products = set()  # Set to track already seen products

        for product_list in product_lists:
            for product in product_list:
                if len(selected_products) < total_limit and product not in seen_products:
                    selected_products.append(product)
                    seen_products.add(product)

            if len(selected_products) >= total_limit:
                break  # Stop once we reach the total limit

        return selected_products


    def predict_next_purchases(self, similar_vectors):  # "arr_search*"
        # predicted_embeddings : [10000, 1, 384]
        # customers : (10000,)
        # products_lists : (10000,)
        # target_embeddings : (10000, 1, 384)
        results = self.loop_over_response(similar_vectors, self.sequential_search)
        return results

    @staticmethod
    def average_hitrate_at_10(recommendations_list: list, actual_purchases_list: list) -> float:

        total_hitrate = 0.0
        num_customers = len(recommendations_list)

        for recommendations, actual_purchases in zip(recommendations_list, actual_purchases_list):
            # Ensure recommendations are unique and limited to top 10
            k = 10
            recommendations = list(dict.fromkeys(recommendations))[:k]

            hits = sum(1 for rec in recommendations if rec in actual_purchases)
            print(hits)
            pdb.set_trace()
            denominator = min(len(actual_purchases), k)

            if denominator > 0:
                total_hitrate += hits / denominator
            else:
                total_hitrate += 0.0

        return total_hitrate / num_customers if num_customers > 0 else 0.0

    def post_evaluation(self, similar_vectors, target_embeddings):
        results = list()
        # Flatten the Qadrant original response, because it cames in batches.
        response = [element for batch in similar_vectors for element in batch]

        for query_response in response:
            product_lists = []
            for scored_point in query_response.points:
                vector = scored_point.vector
                results.append(vector)

            #results.append(calculation_method(product_lists))
            # pdb.set_trace()


        results = np.array(results)
        results = np.expand_dims(results, axis=1)

        results_reshaped = results.squeeze(axis=1)  # Shape becomes (10000, 200)
        target_embeddings_reshaped = target_embeddings.squeeze(axis=1)  # Shape becomes (10000, 200)

        cosine_sim = cosine_similarity(results_reshaped, target_embeddings_reshaped)
        cosine_sim_diag = np.diag(cosine_sim)

        cosine_difference = 1 - cosine_sim_diag
        mean_cosine_difference = cosine_difference.mean()

        print("Mean Cosine Difference:", mean_cosine_difference)



        return results

def main():
    parser = argparse.ArgumentParser(description="Qadrant API caller")
    parser.add_argument('--query', type=str, default=None, help="Query Qadrant for similarity search and save to disk.")
    parser.add_argument('--load', type=str, default=None, help="Load Qadrant query response from disk.")

    args = parser.parse_args()

    start_time = time.time()
    evaluator = EvaluateModel()

    model = evaluator.load_model()
    test_dataloader, customers, products_lists, target_embeddings = evaluator.load_data()

    mse_loss, cosine_loss, predicted_embeddings = evaluator.evaluate(test_dataloader, model)
    evaluator.logger.info(f"Average MSE Loss: {mse_loss}, Average Cosine Similarity Loss: {cosine_loss}, Training Loss: {evaluator.training_val_loss}")

    similar_vectors = None

    if args.query:
        predicted_embeddings = predicted_embeddings.numpy()
        predicted_embeddings = np.squeeze(predicted_embeddings, axis=1)
        similar_vectors = evaluator.get_similar_embeddings_to_disk(predicted_embeddings, filter=customers) # , customers, products_lists, target_embeddings)

    if args.load:
        similar_vectors = evaluator.get_similar_embeddings_from_disk(args.load)
        evaluator.post_evaluation(similar_vectors, target_embeddings)


        results = evaluator.predict_next_purchases(similar_vectors)

        #results = evaluator.refine_products(results,)

        #target_embeddings = target_embeddings.squeeze(axis=1)  # Shape: [10000, 384]
        pdb.set_trace()

        metric = evaluator.average_hitrate_at_10(results, products_lists)
        print("Top10", metric)

    end_time = time.time()
    evaluator.logger.info(f"Train Execution time {(end_time - start_time):.2f} seconds.")

if __name__ == "__main__":
    main()
