from config import PAD_IDX

import os

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

# base class
class Trainer():
    def __init__(self, n_epochs, save_checkpoints=True):
        self.n_epochs = n_epochs
        self.device = self.get_default_device()
        self.save_checkpoints = save_checkpoints
        self.writer = SummaryWriter(f'./runs')

    def set_device(self, device):
        self.device = device

    def get_default_device(self):
        # check cuda
        if torch.cuda.is_available():
            return torch.device("cuda")
        # check mac metal-shaders (m series) gpu
        if torch.backends.mps.is_available():
            return torch.device("mps")
        # devices above are not available, use cpu
        return torch.device("cpu")

    # model: torch.nn.Module
    # data: torch.DataLoader
    def fit(self, model, train_dataloader, valid_dataloader):
        # get data
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        # get model
        self.model = model.to(self.device)

        print("--- Starting training for", self.model.name)

        # run training loop
        for epoch in range(self.n_epochs):
            print(f"----- Epoch {epoch} -----")

            train_loss, train_acc = self.fit_epoch(epoch)
            valid_loss, valid_acc = self.valid_epoch(epoch)
            
            if self.save_checkpoints: model.save_model()

        print("---\nFinished training.")

    # return the epoch loss
    def fit_epoch(self, epoch) -> float:
        self.model.train()
        self.model = self.model.to(self.device)

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        n_batches = len(self.train_dataloader)

        epoch_losses = []
        epoch_accs = []

        # get the batch from the dataloader
        for i, batch in enumerate(tqdm(self.train_dataloader)):

            # send the data to the same device as the model
            x, y, x_l, _ = batch
            x = x.to(self.device)
            y = y.to(self.device)

            # reset the gradients
            self.model.optim.zero_grad()

            # make the prediction
            out = self.model(x, x_l, y)
            out = out.view(-1, out.size(-1))
            y = y.view(-1)

            # calculate the loss
            loss = self.model.criterion(out, y)
            batch_loss = loss.item()

            # calculate the accuracy
            pad_mask = (y != PAD_IDX)
            correct = (torch.max(out, dim=1)[1][pad_mask] == y[pad_mask]).sum().item()
            batch_accuracy = (correct / pad_mask.sum().item()) * 100.0 if pad_mask.sum().item() > 0 else 0.0

            # step backward to calculate gradients (back propagation)
            loss.backward()

            # optimize the weights based on the gradients
            self.model.optim.step()

            # data logging
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            epoch_losses.append(batch_loss)
            epoch_accs.append(batch_accuracy)
            self.model.train_losses.append(batch_loss)
            self.model.train_accs.append(batch_accuracy)

            # add data to tensorboard
            step = epoch * n_batches + i
            self.writer.add_scalars(f"Loss({self.model.name})", {'train':batch_loss}, step)
            self.writer.add_scalars(f"Accuracy({self.model.name})", {'train':batch_accuracy}, step)

            print(f"[Valid] Batch: {i}/{n_batches}")
            print(f"\tTrain Loss: {batch_loss:.6f}")
            print(f"\tTrain Acc: {batch_accuracy:.6f}")

            # clean up
            del correct
            del pad_mask
            del loss
            del out
            del y
            del x
            torch.cuda.empty_cache()
            
            if i % 100 == 0 and i > 0:
                if self.save_checkpoints: self.model.save_model()
                plt.title(self.model.get_name_with_hyper_params() + " Training Performance")
                plt.grid()
                plt.xlabel("Batch")
                plt.plot(self.model.train_losses, c='b', label="Loss")
                plt.plot(self.model.train_accs, c='y', label="Acc %")
                plt.legend()
                plt.savefig("./checkpoints/" + self.model.name + "/train_plot_full.png")
                plt.close()
        # end of loop

        epoch_loss = epoch_loss / n_batches
        print(f"[Train] Average Epoch Loss: {epoch_loss:.6f}")
        epoch_accuracy = (epoch_accuracy / n_batches) * 100.0
        print(f"[Train] Average Epoch Accuracy: {epoch_accuracy:.6f}")

        return epoch_loss, epoch_accuracy

    # based on the training epoch method above, but without training (no .backward(), etc)
    def valid_epoch(self, epoch) -> float:
        self.model.eval()
        with torch.no_grad():

            epoch_loss = 0.0
            epoch_accuracy = 0.0
            n_batches = len(self.valid_dataloader)
            
            epoch_losses = []
            epoch_accs = []
            
            # load batch using dataloader
            for i, batch in enumerate(tqdm(self.valid_dataloader)):

                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                # make the prediction
                out = self.model(x, torch.tensor([x.shape[0]]), y)
                out = out.view(-1, out.size(-1))
                y = y.view(-1)

                # calculate the loss
                loss = self.model.criterion(out, y)
                batch_loss = loss.item()

                # calculate accuracy
                pad_mask = (y != PAD_IDX)
                correct = (torch.max(out, dim=1)[1][pad_mask] == y[pad_mask]).sum().item()
                batch_accuracy = (correct / pad_mask.sum().item()) * 100.0 if pad_mask.sum().item() > 0 else 0.0

                # log data
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                epoch_losses.append(batch_loss)
                epoch_accs.append(batch_accuracy)
                self.model.valid_losses.append(batch_loss)
                self.model.valid_accs.append(batch_accuracy)

                # add data to tensorboard
                step = epoch * n_batches + i
                self.writer.add_scalars(f"Loss({self.model.name})", {'valid':batch_loss}, step)
                self.writer.add_scalars(f"Accuracy({self.model.name})", {'valid':batch_accuracy}, step)

                print(f"[Valid] Batch: {i}/{n_batches}")
                print(f"\tValid Loss: {batch_loss:.6f}")
                print(f"\tValid Acc: {batch_accuracy:.6f}")

                # clean up
                del correct
                del pad_mask
                del loss
                del out
                del y
                del x
                torch.cuda.empty_cache()
                
                if i % 100 == 0 and i > 0:
                    plt.title(self.model.get_name_with_hyper_params() + " Validation Performance")
                    plt.grid()
                    plt.xlabel("Batch")
                    plt.plot(self.model.valid_losses, c='b', label="Loss")
                    plt.plot(self.model.valid_accs, c='y', label="Acc %")
                    plt.legend()
                    plt.savefig("./checkpoints/" + self.model.name + "/valid_plot_full.png")
                    plt.close()
            # end of loop

            epoch_loss = epoch_loss / n_batches
            print(f"[Valid] Average Epoch Loss: {epoch_loss:.6f}")
            epoch_accuracy = (epoch_accuracy / n_batches) * 100.0
            print(f"[Valid] Average Epoch Accuracy: {epoch_accuracy:.6f}")
            
            return epoch_loss, epoch_accuracy



