import torch
from torch.utils.tensorboard import SummaryWriter

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
        self.test_dataloader = test_dataloader
        # get model
        self.model = model.to(self.device)

        print("--- Starting training for", self.model.name)

        # run training loop
        for epoch in range(self.n_epochs):
            print(f"----- Epoch {epoch} -----")

            self.fit_epoch(epoch)
            self.valid_epoch(epoch)
            
            if self.save_checkpoints:
                model.save_model()
        print("---\nFinished training.")

    # return the epoch loss
    def fit_epoch(self, epoch) -> float:
        epoch_loss = 0.0

        n_batches = len(self.train_dataloader)

        self.model = self.model.to(self.device)
        # may use model.train() to ensure the gradients are able to be updated
        self.model.train()

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

            # step backward to calculate gradients (back propagation)
            loss.backward()

            # optimize the weights based on the gradients
            self.model.optim.step()

            # data logging ---
            epoch_loss += batch_loss
            
            self.model.train_losses.append(batch_loss)

            # add data to tensorboard
            step = epoch * n_batches + i
            self.writer.add_scalars(f"Loss({self.model.name})", {'train':batch_loss}, step)

            print(f"Batch: {i}/{n_batches}")
            print(f"\tTrain Loss: {batch_loss:.6f}")

            # clean up
            del loss
            del out
            del y
            del x
            torch.cuda.empty_cache()
            
            if i % 3000 == 0:
                self.model.save_model()

        epoch_loss = epoch_loss / n_batches
        print(f"[Train] Average Loss: {epoch_loss:.6f}")

        return epoch_loss

    # based on the training epoch method above, but without training (no .backward(), etc)
    # return the valid epoch loss
    def valid_epoch(self, epoch) -> float:
        epoch_loss = 0.0
        n_batches = len(self.test_dataloader)
        
        # alternative to model.eval()
        # not necessary to have both
        self.model = self.model.to(torch.device("cpu"))
        with torch.no_grad():
            self.model.eval()

            # load batch using dataloader
            for i, batch in enumerate(self.test_dataloader):
                x, y, x_l, _ = batch
                
                # make prediction
                # use None for y
                out = self.model(x, x.shape(1), None)

                # reshape output
                out = out.view(-1, out.size(-1))
                y = y.view(-1)

                # calculate loss
                loss = self.model.loss(out, y)
                batch_loss = loss.detach().cpu()

                epoch_loss += batch_loss

                self.model.test_losses.append(batch_loss)

                step = epoch * n_batches + i
                self.writer.add_scalars(f"Loss({self.model.name})", {'valid':batch_loss}, step)
                print(f"\tValid Loss: {batch_loss:.6f}")

            epoch_loss = epoch_loss / n_batches
            
            print(f"[Valid] Average Loss: {epoch_loss:.6f}")
            
            return epoch_loss



