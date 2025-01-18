import torch
import numpy as np
from torch.utils.data import Dataset
import pdb
import os
import pickle

class AggregateData():
    def __init__(self, X1, X2, y, products_to_embeddings_dict, unique_sequences_id_to_embedding_dict):

        self.X1 = X1
        self.X2 = X2
        self.y = torch.tensor(y, dtype=torch.float32)
        self.products_to_embeddings_dict = products_to_embeddings_dict
        self.unique_sequences_id_to_embedding_dict = unique_sequences_id_to_embedding_dict

        #self.base_path = "/content/drive/MyDrive/MLDM-PROJ/"
        #self.base_path = "/home/pv10123z/mldm-project/"
        self.base_path = "/Users/Documents/mldm-project/"
        self.data_dir = "data/train"

        self.X1_path = os.path.join(self.base_path, self.data_dir, "X1_prob_V200_P1.pkl")
        self.X2_path = os.path.join(self.base_path, self.data_dir, "X2_prob_V200_P1.pkl")
        self.y_path = os.path.join(self.base_path, self.data_dir, "y_prob_V200_P1.pkl")

        self.products_to_embeddings_path= os.path.join(self.base_path, self.data_dir, "products_embeddings_dict.pkl")
        self.unique_sequences_path= os.path.join(self.base_path, self.data_dir, "unique_sequences_V200_p1.pkl")
        self.unique_sequences_id_to_embedding_path = os.path.join(self.base_path, self.data_dir, "unique_sequences_id_to_embedding.pkl")

    def get_data_from_pkl(self):
        # Load historic transactions
        with open(self.X1_path, 'rb') as f:
            X1 = pickle.load(f)

        # Load candidate product
        with open(self.X2_path, 'rb') as f:
            X2 = pickle.load(f)

        # Load labels
        with open(self.y_path, 'rb') as f:
            y = pickle.load(f)

        # Load products to embeddings dict
        with open(self.products_to_embeddings_path, 'rb') as f:
            products_embeddings_dict = pickle.load(f)

        # Load unique sequences id to embedding dict
        with open(self.unique_sequences_id_to_embedding_path, 'rb') as f:
            unique_sequences_id_to_embedding_dict = pickle.load(f)


        X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
            X1, X2, y, test_size=0.2, random_state=1
        )

        return X1_train, X1_val, X2_train, X2_val, y_train, y_val, products_embeddings_dict, unique_sequences_id_to_embedding_dict

    @staticmethod
    def save_to_pkl(objects_dict, output_dir_path):

        for name, obj in objects_dict.items():
            save_path = os.path.join(output_dir_path, f"{name}.pkl")

            with open(save_path, 'wb') as f:
                pickle.dump(obj, f)

    def aggregate_data(self):
        X1_data = list()
        X2_data = list()

        for element in self.X1:
            X1 = self.unique_sequences_id_to_embedding_dict[element]
            X1_data.append(X1)

        for element in self.X2:
            X2 = self.products_to_embeddings_dict[self.X2]
            X2_data.append(X2)
        pdb.set_trace()

        self.save_to_pkl({"X1_FULL": X1_data, "X2_FULL": X2_data}, "/Users/Documents/mldm-project/data/aggregated/")

def main():


    trainer = AggregateData()

    train_dataloader, val_dataloader = trainer.load_data()


    end_time = time.time()
    trainer.logger.info(f"Train Execution time {(end_time - start_time):.2f} seconds.")

if __name__ == "__main__":
    main()