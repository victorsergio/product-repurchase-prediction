import pdb

#from transformer import Transformer
#from transformer_dataset import TransformerDataset
from ranking_model import RankingModel
from ranking_model_dataset import RankingModelDataset

from torch.utils.data import DataLoader
import torch
import logging
import transformers as hftransformers
import wandb
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time
import configparser
from utils import save_to_json

from utils import setup_logger


class TrainRankingModel:

    def __init__(self):

        self.base_path = "/content/drive/MyDrive/MLDM-PROJ/"
        
        self.data_dir = "data/train"
        self.config_path = os.path.join(self.base_path, "config.ini")
        self.save_model_dir = "out/ranking_model"
        self.start_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.save_model_path = os.path.join(self.base_path, self.save_model_dir, self.start_timestamp)
        self.reload_model_path = os.path.join(self.base_path, self.save_model_dir,
                                              "07-01-2025_13-20-49")  # used for retraining a model
        self.log_dir = "logs"
        self.log_path = os.path.join(self.base_path, self.log_dir, f"{self.start_timestamp}_training_ranking.log")

        # Create a new directory for this experiment run
        os.makedirs(self.save_model_path, exist_ok=True)

        self.logger = setup_logger(self.log_path, to_console=False, logger_name=__name__)

        self.model_config = {
            "embedding_dim": 200,
            "hidden_dim": 64,
            "fc_dim": 128,
        }

        self.training_config = {
            "epochs": 200,
            "batch_size": 128,
            "save_model_path": self.save_model_path,
            "device": "cuda",
            "learning_rate": 0.01
        }

        # Save current experiment configs to files
        save_to_json({"model_config": self.model_config, "training_config": self.training_config}, self.save_model_path)

        #self.historic_path = os.path.join(self.base_path, self.data_dir, "X_V200_P1.pkl")
        #self.next_path = os.path.join(self.base_path, self.data_dir, "y_V200_P1.pkl")

        self.X1_path = os.path.join(self.base_path, self.data_dir, "X1_prob_V200_P1.pkl")
        self.X2_path = os.path.join(self.base_path, self.data_dir, "X2_prob_V200_P1.pkl")
        self.y_path = os.path.join(self.base_path, self.data_dir, "y_prob_V200_P1.pkl")

        self.products_to_embeddings_path= os.path.join(self.base_path, self.data_dir, "products_embeddings_dict.pkl")
        self.unique_sequences_path= os.path.join(self.base_path, self.data_dir, "unique_sequences_V200_p1.pkl")
        self.unique_sequences_id_to_embedding_path = os.path.join(self.base_path, self.data_dir, "unique_sequences_id_to_embedding.pkl")

        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)

        self.wandb_key = self.config['wandb']['key']
        wandb.login(key=self.wandb_key)
        self.wandb_project_name = "mldm-project-ranking"
        wandb.init(project=self.wandb_project_name, name=self.start_timestamp)

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

        # Load unique sequences dict
        #with open(self.unique_sequences_path, 'rb') as f:
        #    unique_sequences = pickle.load(f)

        # Load unique sequences id to embedding dict
        with open(self.unique_sequences_id_to_embedding_path, 'rb') as f:
            unique_sequences_id_to_embedding_dict = pickle.load(f)


        X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
            X1, X2, y, test_size=0.2, random_state=1
        )


        self.logger.info(f"Reading data from .pkl : SUCCESS ")
        return X1_train, X1_val, X2_train, X2_val, y_train, y_val, products_embeddings_dict, unique_sequences_id_to_embedding_dict

    def load_data(self):

        X1_train, X1_val, X2_train, X2_val, y_train, y_val, products_embeddings_dict,unique_sequences_id_to_embedding_dict = self.get_data_from_pkl()

        # A 2 dimension vector is required for the labels (samples, 1)
        y_train = np.expand_dims(y_train, axis=1)
        y_val = np.expand_dims(y_val, axis=1)

        # Create Datasets
        train_dataset = RankingModelDataset(X1_train, X2_train, y_train, products_embeddings_dict, unique_sequences_id_to_embedding_dict )
        val_dataset = RankingModelDataset(X1_val, X2_val, y_val, products_embeddings_dict, unique_sequences_id_to_embedding_dict )

        # Create Dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.training_config["batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.training_config["batch_size"], shuffle=False)

        # Test data_loader
        # Create an iterator

        #dataloader_iter = iter(train_dataloader)
        #batch = next(dataloader_iter)
        #in_1, in_2, label = batch
        #pdb.set_trace()

        self.logger.info(f"Creating dataloaders : SUCCESS ")
        return train_dataloader, val_dataloader

    def train(self, train_dataloader, val_dataloader, retraining=False):

        self.logger.info(f"Train starts.")

        device = self.training_config["device"]
        epochs = self.training_config["epochs"]
        save_model_path = self.training_config["save_model_path"]
        learning_rate = self.training_config["learning_rate"]

        criterion = torch.nn.BCELoss()
        min_val_loss = float('inf')
        best_epoch = -1
        start_epoch = 0

        device = torch.device(device)

        model = RankingModel(self.model_config).double().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Continue training an existing model
        if retraining:
            # Load the checkpoint
            checkpoint = torch.load(os.path.join(self.reload_model_path, "checkpoint.pth"))

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            min_val_loss = checkpoint["val_loss"]

            self.logger.info(f"Checkpoint {self.reload_model_path} loaded for retraining : SUCCESS ")

        scheduler = hftransformers.get_cosine_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=(len(train_dataloader) * 10),
                                                                   num_training_steps=len(train_dataloader) * epochs)

        for epoch in range(start_epoch, start_epoch + epochs):

            train_loss = 0
            val_loss = 0

            # Training Loop
            model.train()
            for _input_X1, _input_X2, target, *_ in train_dataloader:
                optimizer.zero_grad()

                src_1 = _input_X1.double().to(device)
                src_2 = _input_X2.double().to(device)
                target = target.double().to(device)

                prediction = model(src_1, src_2)

                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().item()

            # Validation Loop
            model.eval()
            with torch.no_grad():
                for _input_X1, _input_X2, target, *_ in val_dataloader:
                    src_1 = _input_X1.double().to(device)
                    src_2 = _input_X2.double().to(device)
                    target = target.double().to(device)

                    prediction = model(src_1, src_2)
                    loss = criterion(prediction, target)
                    val_loss += loss.item()

            # Compute average losses
            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)

            # Log and save the best model based on validation loss
            if val_loss < min_val_loss:
                best_epoch = epoch
                min_val_loss = val_loss

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss
                }, os.path.join(save_model_path, "checkpoint.pth"))

            self.logger.info(f"Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
            wandb.log({"epoch_": epoch, "training_loss": train_loss, "validation_loss": val_loss}, step=epoch)

            scheduler.step()

        # Log model to wandb cloud
        wandb.save(os.path.join(save_model_path, "checkpoint.pth"), base_path=save_model_path)
        wandb.save(os.path.join(save_model_path, "model_config.json"), base_path=save_model_path)
        wandb.save(os.path.join(save_model_path, "training_config.json"), base_path=save_model_path)

        self.logger.info(
            f"Best model saved to: {save_model_path} directory. Min Val Loss: {min_val_loss} Best Epoch: {best_epoch}")
        self.logger.info(f"Train finished: SUCCESS ")
        return True


def main():
    start_time = time.time()

    trainer = TrainRankingModel()

    train_dataloader, val_dataloader = trainer.load_data()
    trainer.train(train_dataloader, val_dataloader, retraining=True)

    end_time = time.time()
    trainer.logger.info(f"Train Execution time {(end_time - start_time):.2f} seconds.")

if __name__ == "__main__":
    main()
