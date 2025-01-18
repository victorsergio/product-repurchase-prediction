import pdb

import pandas as pd
import os
import re
import time
import unicodedata
import numpy as np
import pickle
import logging
from datetime import datetime
import fasttext
from numpy.ma.core import product
from tqdm import tqdm
from utils import setup_logger

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings
from qdrant_api import QdrantAPI

from scipy.spatial.distance import cosine

from collections import Counter
import matplotlib.pyplot as plt


# Define a custom callback to track Word2Vec loss
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            logging.info(f'Epoch {self.epoch} completed with loss: {loss}')
        else:
            logging.info(f'Epoch {self.epoch} completed with loss: {loss-self.loss_previous_step}')
        self.epoch += 1
        self.loss_previous_step = loss

class DataPreprocessing:
    def __init__(self):
        self.start_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        #self.base_path = "/content/drive/MyDrive/MLDM Project/src_v2/"
        self.base_path = "/Users/Documents/mldm-project/"
        # base_path = "/home/pv10123z/mldm-project/"
        self.data_dir = "data/raw_data/"
        self.data_path = os.path.join(self.base_path, self.data_dir)
        self.output_dir = "data/train"
        self.log_dir = "logs"

        self.embedding_model_dir = "fasttext"


        self.common_file_name = "train_data_part_"
        self.products_file_name = "products_data.csv"
        self.test_data = "test_data.csv"  # Test data, next future order

        self.output_dir_path = os.path.join(self.base_path, self.output_dir)
        self.historic_filename = "historic_V200_P1"
        #self.next_filename = "y_V200_P1"
        self.X_filename = "X_V200_P1"
        self.y_filename = "y_V200_P1"

        self.X1_prob_filename = "X1_prob_V200_P1"
        self.X2_prob_filename = "X2_prob_V200_P1"
        self.y_prob_filename = "y_prob_V200_P1"
        self.unique_sequences_filename = "unique_sequences_V200_P1"
        self.unique_sequences_id_to_embedding_filename = "unique_sequences_id_to_embedding"

        self.test_output_dir_path = os.path.join(self.base_path, "data/test")

        #self.customers_filename = "customers_T0_V384_P1"
        #self.products_lists_filename = "products_lists_T0_V384_P1"


        self.log_path = os.path.join(self.base_path, self.log_dir, f"{self.start_timestamp}_data_preprocessing.log")
        self.logger = setup_logger(self.log_path, to_console=True, logger_name=__name__)

        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s",
            level=logging.INFO  # Use DEBUG for more details
        )

        self.n_files = 2
        self.fasttext_embedding_size = 100
        self.fasttext_window = 10   # Max number of product_id inside a transaction observable for context
        self.fasttext_epochs = 60   # 92 last

        self.sequence_length = 3

        self.products_embeddings_dict = dict()
        self.negative_products_dict = dict()
        self.embeddings_to_products_dict = dict()
        self.negative_products_ids_dict = dict()

        self.products_to_vector_dict = dict()
        self.vector_to_product_dict = dict()


    def load_embedded_data_parquet(self):
        df = pd.read_parquet(os.path.join(self.base_path, self.output_dir, self.historic_filename + ".parquet"))

        customers_ids = df.customer_id
        target_embeddings = df.transaction_embedding
        products_lists = df.products
        transactions_ids = df.transaction_id
        dates = df.date

        return customers_ids, products_lists, target_embeddings, transactions_ids, dates


    @staticmethod
    def save_to_pkl(objects_dict, output_dir_path):

        for name, obj in objects_dict.items():
            save_path = os.path.join(output_dir_path, f"{name}.pkl")

            with open(save_path, 'wb') as f:
                pickle.dump(obj, f)

    @staticmethod
    def load_data(n_files, data_path, common_file_name : str, products_file_name: str, test_data: str):

        historic_data = list()

        # Load historic train data
        for i in range(0, n_files):
            historic_data.append(pd.read_csv(os.path.join(data_path, common_file_name + str(i + 1) + ".csv")))

        # Join all the training data
        historic_df = pd.concat(historic_data, ignore_index=True)

        # Load product table
        product_df = pd.read_csv(os.path.join(data_path, products_file_name), low_memory=False)

        # Load next purchase table
        next_purchase_df = pd.read_csv(os.path.join(data_path, test_data))

        return historic_df, next_purchase_df, product_df

    @staticmethod
    def remove_accents(input_str):
        return ''.join(
            c for c in unicodedata.normalize('NFD', input_str)
            if unicodedata.category(c) != 'Mn'
        )

    # Define a function to pre-process text product descriptions
    @staticmethod
    def process_description(s, min_word_length, n_words):

        # Normalize text to remove accents
        s = DataPreprocessing.remove_accents(s)

        # Replace dots in words with spaces to split subwords
        s = re.sub(r'\.', ' ', s)

        # Remove punctuation, special symbols
        s = re.sub(r'[^\w\s]', '', s)

        # filter word length
        words = [
            word for word in s.split()
            #if len(word) >= min_word_length and not re.search(r'\d', word)
        ]

        # Get n words to represent a product description
        #if n_words > 0:
        #    words = words[:n_words]

        # Join the words into a single string
        processed = ' '.join(words)

        return processed

    @staticmethod
    def get_last_n_transactions(df, n=3):

        # df = df.sort_values(by=['date', 'transaction_id'])
        df = df.sort_values(by=['date'])

        # Get the last n unique transaction_id for each customer_id
        last_n_transactions = (
            df.groupby('customer_id')['transaction_id']
            .apply(lambda x: x.drop_duplicates().tail(n))
            .reset_index(drop=True)
        )

        # Filter the original DataFrame to get all products from last transactions
        result_df = df[df['transaction_id'].isin(last_n_transactions)]

        return result_df

    @staticmethod
    def join_products(df):
        ''' This function returns a dataframe that contains: A row for each customer_id,
            with a string of product_ids
        '''
        temp = df.copy()
        # Add a date column, for the next purchase, because in the validation set is missing
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime('2024-01-01')

        # Sorting of products inside transactions and concatenation as a single object
        df = df.groupby(['customer_id', 'transaction_id']).agg(
            # Sort product_id alphabetically within each transaction
            products=('product_id', lambda x: ' '.join(sorted(map(str, x)))),  # Concatenate product_ids as a string
            date=('date', 'first')
        ).reset_index()

        # Order all the transactions for each client from oldest to newest
        df = df.sort_values(by=['customer_id', 'date']).reset_index(drop=True)

        return df


    def compute_transaction_embedding(self, product_list, model):
        products = product_list.split()
        #products = list(set(products))



        # Get the embedding for each product and aggregate them (e.g., mean)
        #embeddings = [model.get_word_vector(product) for product in products]

        embeddings = [self.products_to_vector_dict.get(product) for product in products if product in self.products_to_vector_dict]

        # Store the embeddings for all products
        #new_products = {product: embedding for product, embedding in zip(products, embeddings)}
        #self.products_embeddings_dict.update(new_products)

        #for product, embedding in zip(products, embeddings):
        #    self.products_embeddings_dict[product] = embedding
        #    self.embeddings_to_products_dict[tuple(embedding)] = product

        agg = np.mean(embeddings, axis=0) if embeddings else np.zeros(self.fasttext_embedding_size)
        return agg

    def test_agg(self):
        a = 'Product_63613 Product_76625'
        b = 'Product_26523 Product_32349'

        while True:
            r1 = self.compute_transaction_embedding(a, None)
            r2 = self.compute_transaction_embedding(b, None)

            cosine_similarity = 1 - cosine(r1, r2)
            print("Cosine Similarity:", cosine_similarity)
            pdb.set_trace()


    def transaction_to_embedding(self, df, model):
        tqdm.pandas(desc="Embedding Transactions")
        df['transaction_embedding'] = df['products'].progress_apply(lambda x: self.compute_transaction_embedding(x, model))
        # Convert embeddings to a list for storing
        df['transaction_embedding'] = df['transaction_embedding'].apply(list)

        return df

    #def get_negative_sample(self, embedding):


    #    negative = qdrant_api.get_less_similar(embedding)

    #    return negative

    def calculate_negative_samples(self):
        qdrant_api = QdrantAPI(
            base_path=self.base_path,
            data_name="products_embeddings_dict.pkl",
            collection_name="products_embeddings_fasttext",
            vector_size=self.fasttext_embedding_size
        )

        products_keys = list(self.products_embeddings_dict.keys())
        products_values = list(self.products_embeddings_dict.values())
        products_values = np.stack(products_values) # Convert to numpy (n_vectors, dimension)

        query_response = qdrant_api.perform_similarity_search(products_values, top_k=100)
        # Get rid of the batches in the query response
        query_response = [item for sublist in query_response for item in sublist]

        # Find the least similar vector for each QueryResponse
        least_similar_vectors = []
        least_similar_products = []

        for response in tqdm(query_response, desc="Processing query responses"):

            points = response.points

            if not points:
                least_similar_vectors.append(None)
                continue

            # Sort the points by score (ascending)
            sorted_points = sorted(points, key=lambda point: point.score)

            # Get the vector with the lowest similarity (least similar)
            least_similar = sorted_points[0].vector  # First result is the least similar
            least_similar_product = "Product_"+str(sorted_points[0].id) # Recover the prefix

            least_similar_vectors.append(least_similar)
            least_similar_products.append(least_similar_product)

        # Create a new dictionary with the negative samples
        # key: product_id, value: the embedding of the most dissimilar product in the database
        self.negative_products_dict = {key: value for key, value in zip(products_keys, least_similar_vectors)}
        self.negative_products_ids_dict = {key: value for key, value in zip(products_keys, least_similar_products)}

        self.save_to_pkl({"negative_products_dict": self.negative_products_dict, "negative_products_ids_dict":self.negative_products_ids_dict}, self.output_dir_path)
        self.logger.info(f"Data saved to negative_products_dict.pkl : SUCCESS")



    def create_3D_data_structure(self, data, sequence_length, embedding_dim):
        sequences = []
        targets = []

        sequences_prob = []
        product_prob = []
        label_prob = []

        unique_sequences = {}
        unique_sequences_id_to_embedding ={}
        sequence_counter = 0  # Counter to track the sequence indices

        for customer_id, group in tqdm(data.groupby('customer_id'), desc="Processing 3D data"):
            group = group.sort_values('date')

            embeddings = np.stack(group['transaction_embedding'].values)  # Convert to (num_transactions, embedding_dim)
            products = group['products'].values

            if len(embeddings) < sequence_length + 1:

                embeddings_original_length = len(embeddings)

                padding = np.zeros(((sequence_length + 1) - embeddings_original_length, embedding_dim))
                embeddings = np.vstack([padding, embeddings])

                # Padding for products
                padding_products = np.full((sequence_length + 1) - embeddings_original_length, '', dtype=str)
                products = np.concatenate((padding_products, products))
                #products = padding_products + products
                #products = np.vstack([padding_products, products])

            if len(embeddings) > sequence_length:
                for i in range(len(embeddings) - sequence_length):
                    sequences.append(embeddings[i:i + sequence_length])  # Past transactions
                    targets.append(embeddings[i + sequence_length])  # Next transaction


                    transaction_products = set(str(products[i + sequence_length]).split())
                    # Remove empty strings from the set (padding)
                    #transaction_products = {element for element in transaction_products if element != ''}

                    try:
                        if len(transaction_products) <= 0:
                            pdb.set_trace()
                            raise ValueError("Transaction products set is empty.")
                    except ValueError as e:
                        print(f"Error: {e}")


                    for p in transaction_products:

                        # This part of code, is to optimize the storage of 3 previous embeddings, storing
                        # only references index to them.

                        # Convert the numpy array to a tuple (hashable type)
                        sequence_key = tuple(embeddings[i:i + sequence_length].flatten())  # Use .flatten() to get a 1D array if necessary

                        # Check if the past transaction sequence is already stored
                        if sequence_key not in unique_sequences:
                            # If not stored, add it to the dictionary
                            unique_sequences[sequence_key] = sequence_counter
                            unique_sequences_id_to_embedding[sequence_counter]   = embeddings[i:i + sequence_length] # Reverse search dict

                            sequence_counter += 1

                        # Get the index of the sequence
                        sequence_index = unique_sequences[sequence_key]


                        # Positive Sample
                        sequences_prob.append(sequence_index)  # Store the index/reference, Past transactions
                        #product_embedding = self.products_embeddings_dict[p]
                        product_embedding = p


                        product_prob.append(product_embedding)
                        label_prob.append(1)

                        # Negative Sample
                        sequences_prob.append(sequence_index)  # Store the index/reference, Past transactions

                        product_embedding = self.negative_products_ids_dict[p]
                        #product_embedding = self.negative_products_dict[p]
                        # Additional step to get the Product Id negative
                        #product_embedding = self.embeddings_to_products_dict[tuple(product_embedding)]
                        # Find the key by value
                        # Assuming self.products_embeddings_dict and product_embedding are NumPy arrays
                        #key = next((k for k, v in self.products_embeddings_dict.items() if np.array_equal(v, product_embedding)), None)

                        #product_embedding = key

                        product_prob.append(product_embedding)
                        label_prob.append(0)


        X = np.array(sequences)
        y = np.array(targets)

        X1_prob = np.array(sequences_prob)
        X2_prob = np.array(product_prob)
        y_prob = np.array(label_prob)

        return X, y, X1_prob, X2_prob, y_prob, unique_sequences, unique_sequences_id_to_embedding

    @staticmethod
    def create_sequential_data_structure(historic_df, next_df, embedding_size=200, last_transactions=3):
        # Pre-filter next to only include customers present in historic
        next_filtered = next_df[next_df['customer_id'].isin(historic_df['customer_id'].unique())]

        historic_df = historic_df.sort_values(by=['customer_id', 'date'])
        next_filtered = next_filtered.sort_values(by=['customer_id', 'date'])

        grouped = historic_df.groupby('customer_id')

        input_sequences = []
        target_embeddings = []
        customers_ids = []
        products_lists = []

        for _, next_row in next_filtered.iterrows():  # Assuming only one transaction_id per customer_id exists in the next dataframe
            customer_id = next_row['customer_id']

            # Get the customer's historical transactions
            customer_history = grouped.get_group(customer_id).tail(last_transactions)
            history_embeddings = customer_history['transaction_embedding'].tolist()

            # Pad with zero vectors if fewer than last_transactions
            while len(history_embeddings) < last_transactions:
                history_embeddings.insert(0, np.zeros(embedding_size))

            # Append input and target
            input_sequences.append(history_embeddings)
            target_embeddings.append(next_row['transaction_embedding'])

            customers_ids.append(customer_id)
            products_lists.append(next_row['products']) #Next product ids list

        input_sequences = np.array(input_sequences)  # Shape: (num_customers, n_last_transactions, embedding_size)
        target_embeddings = np.array(target_embeddings)  # Shape: (num_customers, embedding_size)
        customers_ids = np.array(customers_ids)
        products_lists = np.array(products_lists)

        return input_sequences, target_embeddings, customers_ids, products_lists

    def load_embedding_model(self):
        model = fasttext.load_model(os.path.join(self.base_path, self.embedding_model_dir,"fasttext_200.bin"))
        self.logger.info(f"FastText Load : SUCCESS")

        return model

    def train_embedding_w2v_model(self, df):

        transactions = df['products'].apply(lambda x: x.split()).tolist()

        model = Word2Vec(
            sentences=transactions,
            vector_size=self.fasttext_embedding_size,
            window=self.fasttext_window,
            sg=0,  # Skip-gram model (set sg=0 for CBOW)
            negative=20,
            workers=4,
            min_count=700,
            epochs=self.fasttext_epochs,
            compute_loss=True,
            callbacks=[EpochLogger()]
        )

        # Create a dictionary with product IDs as keys and vectors as values
        product_to_vector_dict = {word: model.wv[word] for word in model.wv.index_to_key}

        model.save(os.path.join(self.base_path, self.embedding_model_dir, "word2vec_200.model"))

        # Reverse dictionary, embedding to product_id  tuple(embedding)
        vector_to_product_dict = {tuple(value): key for key, value in product_to_vector_dict.items()}



        self.save_to_pkl({"product_to_vector_dict": product_to_vector_dict, "vector_to_product_dict": vector_to_product_dict}, self.output_dir_path)

        self.logger.info(f"Word2Vec Train Loss: {model.get_latest_training_loss()} : SUCCESS")


    def train_embedding_model(self, df):

        # Save training data to a corpus temp file for training
        df['products'].to_csv(os.path.join(self.embedding_model_dir, "training_corpus.txt"), index=False,
                              header=False)

        model = fasttext.train_unsupervised(
            input=os.path.join(self.embedding_model_dir, "training_corpus.txt"),
            model='skipgram',
            dim=self.fasttext_embedding_size,
            ws=self.fasttext_window,
            minn=0,  # Minimum length of character n-grams (0 to disable subword info)
            maxn=0,  # Maximum length of character n-grams (0 to disable subword info)
            neg=10,  # Number of negative samples
            epoch=self.fasttext_epochs,
            word_ngrams=1,  # Use word-level n-grams (1-gram)
            verbose=2,
            thread = 4,
            minCount=1
        )

        model.save_model(os.path.join(self.base_path, self.embedding_model_dir,"fasttext_200.bin"))

        self.logger.info(f"FastText Train : SUCCESS")

    # Function to remove duplicates while preserving order
    def remove_duplicates(self, product_str):
        return ' '.join(dict.fromkeys(product_str.split()))

    def remove_missing(self, product_str):
        # Split the product string into a list of products
        product_list = product_str.split()

        # Remove duplicates while keeping the order and filtering out products not in the dictionary
        filtered_products = [product for product in product_list if product in self.products_to_vector_dict]

        # Return the filtered products as a space-separated string
        return ' '.join(dict.fromkeys(filtered_products))


    def create_dataframes(self, train=True, calculate=False, upload=False, upload_products=False, calculate_negatives=False, create_3D_dataset=False, calculate_test_data=False):

        # Load Data
        historic_df, next_purchase_df, product_df = DataPreprocessing.load_data(self.n_files, self.data_path, self.common_file_name, self.products_file_name, self.test_data)

        # Filter historic for testing "Household_" between 1 and 100
        #historic_df['customer_number'] = historic_df['customer_id'].str.extract(r'Household_(\d+)', expand=False).astype(float)
        #historic_df = historic_df[(historic_df['customer_number'] >= 1) & (historic_df['customer_number'] <= 100)]
        #historic_df = historic_df.drop(columns=['customer_number'])

        # Filter rows for December 2023
        #historic_df['date'] = pd.to_datetime(historic_df['date'])
        #historic_df = historic_df[(historic_df['date'] >= '2023-12-01') & (historic_df['date'] <= '2023-12-31')]



        # Flatten transaction's products as a single text
        next_purchase_processed = DataPreprocessing.join_products(next_purchase_df)
        historic_processed = DataPreprocessing.join_products(historic_df)


        # Remove duplicates in the product lists
        historic_processed['products'] = historic_processed['products'].apply(self.remove_duplicates)



        # Data for training model, and training embedding model
        #data = pd.concat([historic_processed, next_purchase_processed], ignore_index=True)
        data = historic_processed # Use only the historic, leaving out the validation (2024)

        data['date'] = pd.to_datetime(data['date'])

        self.logger.info(f"Total Transactions found: {len(data)}")

        if train:
            # Train an embedding model from scratch or use train=False to load a pretrained model
            #self.train_embedding_model(data)
            self.train_embedding_w2v_model(data)

        if calculate_test_data:
            #data = pd.concat([historic_processed, next_purchase_processed], ignore_index=True)
            model = self.load_embedding_model()

            # Convert transactions to a embedding representation
            past_processed = self.transaction_to_embedding(historic_processed, model)
            future_processed = self.transaction_to_embedding(next_purchase_processed, model)
            self.logger.info(f"Embedding Calculation : SUCCESS")

            input_sequences, target_embeddings, customers_ids, products_lists = self.create_sequential_data_structure(past_processed, future_processed)

            self.save_to_pkl({"X1_test": input_sequences,
                              "y_test": target_embeddings,
                              "customers_ids_test": customers_ids,
                              "products_ids_test": products_lists
                              }, self.test_output_dir_path)

            self.logger.info(f"Test Data saved to {self.test_output_dir_path} : SUCCESS")

            pdb.set_trace()


        if calculate:
            model = self.load_embedding_model()

            # Load W2V Dictionaries
            with open(os.path.join(self.output_dir_path, "product_to_vector_dict.pkl"), 'rb') as f:
                self.products_to_vector_dict = pickle.load(f)
            with open(os.path.join(self.output_dir_path, "vector_to_product_dict.pkl"), 'rb') as f:
                self.vector_to_product_dict = pickle.load(f)

            #self.test_agg()

            data['products'] = data['products'].apply(self.remove_missing)
            data = data[data['products'] != ''].copy()

            # Convert transactions to a embedding representation
            data_processed = self.transaction_to_embedding(data, model)




            self.logger.info(f"Embedding Calculation : SUCCESS")
            #pdb.set_trace()
            # Save relation product -> embedding
            #self.save_to_pkl({"products_embeddings_dict": self.products_embeddings_dict,
            #                  "embeddings_to_products_dict":self.embeddings_to_products_dict}, self.output_dir_path)
            #self.logger.info(f"Data saved to products_embeddings_dict.pkl : SUCCESS")

            # Save relation transaction -> embedding
            data_processed.to_parquet(
                os.path.join(self.output_dir_path, self.historic_filename + ".parquet"))

            self.logger.info(f"Data saved to {self.historic_filename} : SUCCESS")

        if upload:
            qdrant_api = QdrantAPI(
                base_path=self.base_path,
                data_name=self.historic_filename + ".parquet",
                collection_name="transactions_embeddings_fasttext_small",
                vector_size=self.fasttext_embedding_size
            )

            qdrant_api.send_embeddings_to_qdrant()
            self.logger.info(f"All embedding vectors send to Qdrant server : SUCCESS")

        if upload_products:
            qdrant_api = QdrantAPI(
                base_path=self.base_path,
                #data_name="products_embeddings_dict.pkl",
                data_name = "product_to_vector_dict.pkl",
                collection_name="products_embeddings_fasttext_small",
                vector_size=self.fasttext_embedding_size
            )

            qdrant_api.send_products_to_qdrant()
            self.logger.info(f"All products embedding vectors send to Qdrant server : SUCCESS")

        if calculate_negatives:
            # Load products embeddings
            with open(os.path.join(self.output_dir_path, "products_embeddings_dict.pkl"), 'rb') as f:
                self.products_embeddings_dict = pickle.load(f)

            self.calculate_negative_samples()


        if create_3D_dataset:
            customers_ids, products_lists, target_embeddings, transactions_ids, dates = self.load_embedded_data_parquet()

            # Load products embeddings
            with open(os.path.join(self.output_dir_path, "products_embeddings_dict.pkl"), 'rb') as f:
                self.products_embeddings_dict = pickle.load(f)

            # Load negative products embeddings
            with open(os.path.join(self.output_dir_path, "negative_products_dict.pkl"), 'rb') as f:
                self.negative_products_dict = pickle.load(f)

            # Load reverse embeddings to products dict
            with open(os.path.join(self.output_dir_path, "embeddings_to_products_dict.pkl"), 'rb') as f:
                self.embeddings_to_products_dict = pickle.load(f)

            # Load negative products relationship by id
            with open(os.path.join(self.output_dir_path, "negative_products_ids_dict.pkl"), 'rb') as f:
                self.negative_products_ids_dict = pickle.load(f)


            # Combine without considering the index
            historic_data = pd.concat([customers_ids.reset_index(drop=True),
                            target_embeddings.reset_index(drop=True),
                            dates.reset_index(drop=True),
                            products_lists.reset_index(drop=True)], axis=1)


            # Filter rows selecting only the last 6 months of data
            historic_data = historic_data[(historic_data['date'].dt.year == 2023) & (historic_data['date'].dt.month >= 6) & (historic_data['date'].dt.month <= 12)]

            # Convert the dataframes to a input, target 3D structure
            X, y, X1_prob, X2_prob, y_prob, unique_sequences, unique_sequences_id_to_embedding = self.create_3D_data_structure(historic_data,self.sequence_length,self.fasttext_embedding_size)

            self.save_to_pkl({self.X_filename: X, self.y_filename: y, self.X1_prob_filename: X1_prob,
                              self.X2_prob_filename:X2_prob, self.y_prob_filename:y_prob,
                              self.unique_sequences_filename:unique_sequences,
                              self.unique_sequences_id_to_embedding_filename:unique_sequences_id_to_embedding},  self.output_dir_path)
            self.logger.info(f"3D sequential datasets created. X: {self.X_filename} y: {self.y_filename}: SUCCESS")



def main():
    start_time = time.time()

    data_preprocessor = DataPreprocessing()
    # Create datasets
    data_preprocessor.create_dataframes()

    end_time = time.time()

    #logger.info(f"Files written to: {output_dir_path}")
    #logger.info(f"Input shape: {input_sequences.shape}")
    #logger.info(f"Target shape: {target_embeddings.shape}")
    #logger.info(f"Customers IDs shape: {customers_ids.shape}")
    #logger.info(f"Products Lists shape: {products_lists.shape}")

    data_preprocessor.logger.info(f"Execution time: {end_time - start_time:.6f} seconds. : SUCCESS")

if __name__ == "__main__":
    main()