from transformer import Transformer
from transformer_dataset import TransformerDataset
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


class TrainModel:

    def __init__(self):

        self.base_path = "/home/pv10123z/mldm-project/"
        #self.base_path = "/Users/victor.penaloza/Documents/mldm-project/"
        self.data_dir = "data/train"
        self.config_path = os.path.join(self.base_path, "config.ini")
        self.save_model_dir = "out"
        self.start_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.save_model_path = os.path.join(self.base_path, self.save_model_dir, self.start_timestamp)
        self.reload_model_path = os.path.join(self.base_path, self.save_model_dir,
                                              "12-12-2024_20-41-04")  # used for retraining a model

        self.log_path = os.path.join(self.base_path, self.save_model_dir, self.start_timestamp, "log.txt")

        # Create a new directory for this experiment run
        os.makedirs(self.save_model_path, exist_ok=True)

        logging.basicConfig(filename=self.log_path, filemode='a', level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
        self.logger = logging.getLogger(__name__)

        self.model_config = {
            "n_encoder_layers": 1,
            "input_size": 384,
            "dim_val": 512,
            "dropout_encoder": 0.2,
            "dropout_pos_enc": 0.1,
            "n_heads": 8,
            "num_predicted_features": 384,
            "max_seq_len": 3
        }

        self.training_config = {
            "epochs": 10000,
            "batch_size": 64,
            "save_model_path": self.save_model_path,
            "device": "cuda",
            "learning_rate": 0.001
        }

        # Save current experiment configs to files
        save_to_json({"model_config": self.model_config, "training_config": self.training_config}, self.save_model_path)

        self.historic_path = os.path.join(self.base_path, self.data_dir, "historic_T3_V384_P1.pkl")
        self.next_path = os.path.join(self.base_path, self.data_dir, "next_V384_P1.pkl")

        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)

        self.wandb_key = self.config['wandb']['key']
        wandb.login(key=self.wandb_key)
        self.wandb_project_name = "mldm-project"
        wandb.init(project=self.wandb_project_name, name=self.start_timestamp)

    def get_data_from_pkl(self):
        # Load historic: X
        with open(self.historic_path, 'rb') as f:
            input_sequences = pickle.load(f)

        # Load next future transactions: Y
        with open(self.next_path, 'rb') as f:
            target_embeddings = pickle.load(f)

        input_sequences_train, input_sequences_val, target_embeddings_train, target_embeddings_val = train_test_split(
            input_sequences, target_embeddings, test_size=0.2, random_state=1
        )
        self.logger.info(f"Reading data from .pkl : SUCCESS ")
        return input_sequences_train, target_embeddings_train, input_sequences_val, target_embeddings_val

    def load_data(self):

        input_sequences_train, target_embeddings_train, input_sequences_val, target_embeddings_val = self.get_data_from_pkl()

        # A 3 dimension vector is required for the labels (samples, 1, embedding_size)
        target_embeddings_train = np.expand_dims(target_embeddings_train, axis=1)
        target_embeddings_val = np.expand_dims(target_embeddings_val, axis=1)

        # Create Datasets
        train_dataset = TransformerDataset(input_sequences_train, target_embeddings_train)
        val_dataset = TransformerDataset(input_sequences_val, target_embeddings_val)

        # Create Dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.training_config["batch_size"], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.training_config["batch_size"], shuffle=False)

        self.logger.info(f"Creating dataloaders : SUCCESS ")
        return train_dataloader, val_dataloader

    def train(self, train_dataloader, val_dataloader, retraining=False):

        self.logger.info(f"Train starts.")

        device = self.training_config["device"]
        epochs = self.training_config["epochs"]
        save_model_path = self.training_config["save_model_path"]
        learning_rate = self.training_config["learning_rate"]

        criterion = torch.nn.MSELoss()
        min_val_loss = float('inf')
        best_epoch = -1
        start_epoch = 0

        device = torch.device(device)

        model = Transformer(self.model_config).double().to(device)
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
            for _input, target in train_dataloader:
                optimizer.zero_grad()

                src = _input.double().to(device)
                target = target.double().to(device)

                prediction = model(src, device)

                loss = criterion(prediction, target[:, :, :])
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().item()

            # Validation Loop
            model.eval()
            with torch.no_grad():
                for _input, target in val_dataloader:
                    src = _input.double().to(device)
                    target = target.double().to(device)

                    prediction = model(src, device)
                    loss = criterion(prediction, target[:, :, :])
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

    trainer = TrainModel()

    train_dataloader, val_dataloader = trainer.load_data()
    trainer.train(train_dataloader, val_dataloader, retraining=False)

    end_time = time.time()
    trainer.logger.info(f"Train Execution time {(end_time - start_time):.2f} seconds.")

if __name__ == "__main__":
    main()
