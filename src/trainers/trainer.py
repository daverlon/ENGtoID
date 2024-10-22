import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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
    def fit(self, model, train_dataloader, test_dataloader):
        # get data
        self.train_dataloader = train_dataloader
        # self.test_dataloader = test_dataloader
        # get model
        self.model = model.to(self.device)

        print("--- Starting training for", self.model.name)

        # run training loop
        for epoch in range(self.n_epochs):
            print(f"----- Epoch {epoch} -----")

            self.fit_epoch(epoch)
            # self.valid_epoch(epoch)
            
            if self.save_checkpoints:
                model.save_model()
        print("---\nFinished training.")

    # return the epoch loss
    def fit_epoch(self, epoch) -> float:
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        n_batches = len(self.train_dataloader)

        self.model = self.model.to(self.device)
        # may use model.train() to ensure the gradients are able to be updated
        self.model.train()

        epoch_losses = []
        epoch_accs = []

        # get the batch from the dataloader
        for i, batch in enumerate(self.train_dataloader):

            x, y, x_l, _ = batch

            # send the data to the same device as the model
            x = x.to(self.device)
            y = y.to(self.device)

            self.model.optim.zero_grad()

            # make the prediction
            out = self.model(x, x_l, y)

            # reshape the output
            out = out.view(-1, out.size(-1))

            y = y.view(-1)

            # calculate the loss
            loss = self.model.criterion(out, y)
            batch_loss = loss.detach().cpu()

            # calculate the accuracy
            _, predicted = torch.max(out, dim=1)
            correct = (predicted == y).sum().item()
            batch_accuracy = (correct / y.numel()) * 100.0

            # step backward to calculate gradients (back propagation)
            loss.backward()

            # optimize the weights based on the gradients
            self.model.optim.step()

            # data logging ---
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

            print(f"Batch: {i}/{n_batches}")
            print(f"\tTrain Loss: {batch_loss:.6f}")
            print(f"\tTrain Acc: {batch_accuracy:.6f}")

            # clean up
            del loss
            del out
            del y
            del x
            torch.cuda.empty_cache()
            
            if i % 100 == 0:
                if self.save_checkpoints:
                    self.model.save_model()
                plt.title(self.model.name + " Training set loss: BS=48, LR=0.001, EPOCHS=5")
                plt.grid()
                plt.ylabel("Loss")
                plt.xlabel("Batch")
                plt.plot(self.model.train_losses, c='b')
                plt.plot(self.model.train_accs, c='y')
                plt.savefig("./models/checkpoints/" + self.model.name + "/full_loss_plot.png")
                plt.close()

        epoch_loss = epoch_loss / n_batches
        print(f"[Train] Average Loss: {epoch_loss:.6f}")
        epoch_accuracy = (epoch_accuracy / n_batches) * 100.0
        print(f"[Train] Average Accuracy: {epoch_accuracy:.6f}")

        plt.title(self.model.name + f" Epoch {epoch} training loss")
        plt.grid()
        plt.ylabel("Loss")
        plt.xlabel("Batch")
        plt.plot(epoch_losses, c='b')
        plt.plot(epoch_accs, c='y')
        plt.savefig("./models/checkpoints/" + self.model.name + f"/epoch_{epoch}_loss_plot.png")
        plt.close()

        return epoch_loss

    # based on the training epoch method above, but without training (no .backward(), etc)
    # return the valid epoch loss
    # def valid_epoch(self, epoch) -> float:
    #     valid_loss = 0.0
    #     n_batches = len(self.test_dataloader)
        
    #     # alternative to model.eval()
    #     # not necessary to have both
    #     self.model = self.model.to(torch.device("cpu"))
    #     with torch.no_grad():
    #         self.model.eval()

    #         # load batch using dataloader
    #         for i, batch in enumerate(self.test_dataloader):
    #             # x, y = batch
    #             x, y = batch
    #             x_l = torch.tensor([x.shape[1]])

    #             out = self.model(out, x_l, None)
                
    #             # reshape output
    #             out = out.view(-1, out.size(-1))
    #             y = y.view(-1)

    #             # calculate loss
    #             loss = self.model.loss(out, y)
    #             batch_loss = loss.detach().cpu()

    #             valid_loss += batch_loss

    #             self.model.test_losses.append(batch_loss)

    #             step = epoch * n_batches + i
    #             self.writer.add_scalars(f"Loss({self.model.name})", {'valid':batch_loss}, step)
    #             print(f"\tValid Loss: {batch_loss:.6f}")

    #         valid_loss = valid_loss / n_batches
            
    #         print(f"[Valid] Average Loss: {epoch_loss:.6f}")
            
    #         return epoch_loss



