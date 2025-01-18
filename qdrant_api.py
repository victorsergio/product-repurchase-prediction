import pdb

from qdrant_client import QdrantClient, models
import os
import configparser
from qdrant_client import QdrantClient
import numpy as np
import pickle
import logging
from datetime import datetime
import argparse
import pandas as pd
from tqdm import tqdm
from utils import setup_logger

class QdrantAPI:

    def __init__(self, base_path="/Users/Documents/mldm-project/", data_name="historic_v200_p1.parquet", collection_name="transactions_embeddings_fasttext", vector_size=200 ):
        #self.base_path = "/content/drive/MyDrive/MLDM Project/SLURM/"
        self.base_path = base_path

        self.data_dir = "data/train"
        self.data_name = data_name

        #self.historic_path = os.path.join(self.base_path, self.data_dir, f"historic_{self.data_name}")
        #self.next_path = os.path.join(self.base_path, self.data_dir, f"next_{self.data_name}")
        #self.customers_path = os.path.join(self.base_path, self.data_dir, f"customers_{self.data_name}")
        #self.products_lists_path = os.path.join(self.base_path, self.data_dir, f"products_lists_{self.data_name}")

        self.log_dir = "logs"
        self.start_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.log_path = os.path.join(self.base_path, self.log_dir, f"{self.start_timestamp}_qdrant.log")

        self.logger = setup_logger(self.log_path, to_console=True, logger_name=__name__)

        self.config_path = os.path.join(self.base_path, "config.ini")

        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)

        # Replace with your Qdrant Cloud details
        qdrant_api_url = self.config['qdrant']['url']
        qdrant_api_key = self.config['qdrant']['key']

        # Initialize Qdrant Cloud client
        self.client = QdrantClient(url=qdrant_api_url, api_key=qdrant_api_key, timeout=3600)

        self.collection_name = collection_name

        self.vector_size = vector_size


    def setup_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
            )
        self.logger.info(f"Qdrant collection setup: SUCCESS ")

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

        self.logger.info(f"Reading data from .pkl : SUCCESS ")

        return input_sequences, target_embeddings, customers, products_lists

    def load_data(self):
        df = pd.read_parquet(os.path.join(self.base_path, self.data_dir, self.data_name))



        customers_ids = df.customer_id
        target_embeddings = df.transaction_embedding
        products_lists = df.products
        transactions_ids = df.transaction_id
        dates = df.date

        #target_embeddings = np.expand_dims(target_embeddings, axis=1)

        return customers_ids, products_lists, target_embeddings, transactions_ids, dates


    # Function to prepare batch upsert data using customer_ids as IDs
    def prepare_qdrant_data(self, target_embeddings, customers_ids, products_lists, transactions_ids, dates):
        """
        Prepare the batch of points using Qdrant's Batch class with customer_ids as IDs.
        """

        ids = [int(transaction.replace('Transaction_', '')) for transaction in transactions_ids]  # Strip the 'Household_' prefix
        #vectors = [embedding.tolist() for embedding in target_embeddings]  # Convert numpy arrays to lists
        # Flatten the target_embeddings to 1D vectors (if they are multi-dimensional)
        vectors = [embedding.flatten().tolist() for embedding in target_embeddings]  # Flatten and convert to lists

        dates = dates.dt.strftime('%Y-%m-%d')

        payloads = [
            {"customer_id": customer_id, "products_list": product_list, "date": date}
            for customer_id, product_list, date in zip(customers_ids, products_lists, dates)
        ]

        return ids, vectors, payloads


    @staticmethod
    def chunk_data(ids, vectors, payloads, batch_size):
        """
        Split the data into batches.
        """
        has_payloads = len(payloads) > 0

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size] if has_payloads else []

            yield batch_ids, batch_vectors, batch_payloads


    def upload_data(self, target_embeddings, customers_ids, products_lists, transactions_ids, dates, batch_size=500):
        # Prepare data for upload
        ids, vectors, payloads = self.prepare_qdrant_data(target_embeddings, customers_ids, products_lists,
                                                          transactions_ids, dates)
        self.upload(ids, vectors, payloads, batch_size)

    def upload(self, ids, vectors, payloads, batch_size=500):
        """
        Upload data to Qdrant collection with batching.
        """

        # Calculate total number of batches
        total_batches = (len(vectors) + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc="Uploading to Qdrant", unit="batch") as pbar:
            # Split the data into batches and upload each batch
            for batch_ids, batch_vectors, batch_payloads in self.chunk_data(ids, vectors, payloads, batch_size):
                try:
                    upload_args = {
                        "collection_name": self.collection_name,
                        "ids": batch_ids,
                        "vectors": batch_vectors,
                        "parallel": 1,  # Number of parallel threads for uploading
                        "max_retries": 3,  # Number of retry attempts
                    }

                    if batch_payloads:
                        upload_args["payload"] = batch_payloads

                    # Upload the batch
                    self.client.upload_collection(**upload_args)

                    self.logger.info(f"Successfully uploaded batch of {len(batch_ids)} points to Qdrant.")
                except Exception as e:
                    self.logger.error(f"Failed to upload batch: {e}")
                finally:
                    pbar.update(1)  # Update progress bar

    def send_embeddings_to_qdrant(self):
        self.setup_collection()
        # Load data from disk
        customers_ids, products_lists, target_embeddings, transactions_ids, dates = self.load_data()
        # Upload data
        self.upload_data(target_embeddings, customers_ids, products_lists, transactions_ids, dates)

    def send_products_to_qdrant(self):
        self.setup_collection()

        # Load product->embedding dictionary
        with open(os.path.join(self.base_path, self.data_dir, self.data_name), 'rb') as f:
            products_dict = pickle.load(f)

        # Prepare data to Qdrant
        ids = [int(product.replace('Product_', '')) for product in list(products_dict.keys())]  # Strip the 'Product_' prefix
        vectors = [embedding.flatten().tolist() for embedding in list(products_dict.values())]
        payloads = list()

        # Store data
        self.upload(ids, vectors, payloads, batch_size=500)

    def search_similar_vectors_batch(self, vectors, top_k=10, batch_size=500, filter=None):
        """
        Search for the top_k most similar vectors for each vector in the input array using batch processing.
        """
        search_requests = []
        all_results = []

        # Split input vectors into batches
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            search_requests.clear()  # Clear previous search requests for each new batch


            if(filter is not None):
                batch_filter = filter[i:i + batch_size]

                # Create search requests for each vector in the batch
                for vector, filter_value in zip(batch_vectors, batch_filter):

                    search_requests.append(models.QueryRequest(
                        query=vector.tolist(),  # Pass the vector as the query
                        limit=top_k,
                        with_payload=True,
                        with_vector=True,
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="customer_id",
                                    match=models.MatchValue(value=filter_value)
                                )
                            ]
                        )
                    ))


            else:
                # Create search requests for each vector in the batch
                for vector in batch_vectors:
                    search_requests.append(models.QueryRequest(
                    query=vector.tolist(),  # Pass the vector as the query
                    limit=top_k,
                    with_payload=True,
                    with_vector=True#,
                    #params=models.SearchParams(hnsw_ef=512)
                    ))



                #search_requests.append(models.SearchRequest(vector=vector.tolist(), limit=top_k))

            try:
                # Perform the batch search using search_batch
                results = self.client.query_batch_points(
                    collection_name=self.collection_name,
                    requests=search_requests
                )

                #results = self.client.search_batch(
                #    collection_name=self.collection_name,
                #    requests=search_requests,
                #    with_vectors=True,
                #    with_payload=True,
                #)
                # If successful, add results to the all_results list
                all_results.append(results)
                self.logger.info(f"Search: Successfully processed batch {i // batch_size + 1}")
            except Exception as e:
                # Log any errors that occur during the batch search
                self.logger.error(f"Search: Error in processing batch {i // batch_size + 1}: {str(e)}")
                return None  # Return None if an error occurs

        return all_results

    def perform_similarity_search(self, input_vectors, top_k=10, filter=None, batch_size=500):
        """
        Perform similarity search on a set of input vectors using batch search.
        """
        if input_vectors.shape[1] != self.vector_size:
            pdb.set_trace()
            raise ValueError(f"Input vectors must have shape (num_vectors, {self.vector_size})")

        # Perform the batch search
        similar_vectors = self.search_similar_vectors_batch(input_vectors, top_k=top_k, batch_size=batch_size, filter=filter)

        if similar_vectors is None:
            self.logger.error("One or more batches failed during the similarity search.")
            return None

        self.logger.info("All batches were processed successfully.")
        return similar_vectors


def main():
    parser = argparse.ArgumentParser(description="Qdrant API caller")
    parser.add_argument('--upload', type=str, default=None, help="Upload embeddings to Qdrant")

    args = parser.parse_args()

    if args.upload:
        qdrant_api = QdrantAPI()
        qdrant_api.send_embeddings_to_qdrant()


if __name__ == "__main__":
    main()