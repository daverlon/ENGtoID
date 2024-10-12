import torch
#from torch.utils.tensorboard import SummaryWriter

# base class
class Trainer():
    def __init__(self, n_epochs, save_checkpoints=True):
        self.n_epochs = n_epochs
        self.device = self.get_default_device()
        self.save_checkpoints = save_checkpoints
        # self.writer = SummaryWriter('./runs')

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
        model.to(self.device)
        self.model = model

        print("--- Starting training for", self.model.name)

        # run training loop
        for epoch in range(self.n_epochs):
            print(f"----- Epoch {epoch} -----")

            self.fit_epoch(epoch)
            self.valid_epoch(epoch)
            
            if self.save_checkpoints:
                model.save_model()
        print("---\nFinished training.")

    def fit_epoch(self, epoch):
        epoch_loss = 0.0
        epoch_accuracy = 0

        n_batches = len(self.train_dataloader)

        # may use model.train() to ensure the gradients are able to be updated
        self.model.train()

        # get the batch from the dataloader
        for i, data in enumerate(self.train_dataloader):

            # batch tuple -> x, y (data, label) (for each item in the batch)
            x, y, lengths = data
            print(x.shape, y.shape)

            bs = len(x)

            # send the data to the same device as the model
            x = x.to(self.device)
            y = y.squeeze()
            y = y.to(self.device)

            # zero the gradients (no accumulating the gradients)
            self.model.optim.zero_grad()

            # make the prediction
            out = self.model(x, lengths)

            # calculate the loss
            loss = self.model.loss(out, y)

            # back propagate through the weights to calculate their impact (gradient/derivative) W.R.T the loss
            loss.backward()

            self.model.optim.step()

            # data logging ---
            batch_loss = loss.item()
            score = (out.argmax(dim=1)==y).sum().item()
            batch_accuracy = score /bs * 100.0

            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy # add average accuracy for this batch

            self.model.train_losses.append(batch_loss)
            self.model.train_accs.append(batch_accuracy)

            # add data to tensorboard
            step = epoch * n_batches + i
            self.writer.add_scalars(f"Loss({self.model.name})", {'train':batch_loss}, step)
            self.writer.add_scalars(f"Acc({self.model.name})", {'train':batch_accuracy}, step)

            if i % 3 == 0:
                print(f"Batch: {i}/{n_batches}")
                print(f"\tTrain Loss: {batch_loss:.6f}")
                print(f"\tTrain Acc:  {score}/{bs} --- {batch_accuracy:.3f}%")

        epoch_loss = epoch_loss / n_batches
        epoch_accuracy = epoch_accuracy / n_batches
        print(f"[Train] Average Loss: {epoch_loss:.6f}")
        print(f"[Train] Average Acc: {epoch_accuracy:.3f}%")

        return (epoch_loss, epoch_accuracy)

    # based on the training epoch method above, but without training (no .backward(), etc)
    def valid_epoch(self, epoch):
        epoch_loss = 0.0
        epoch_accuracy = 0
        n_batches = len(self.test_dataloader)
        
        # alternative to model.eval()
        # not necessary to have both
        with torch.no_grad():
            self.model.eval()

            # load batch using dataloader
            for i, data in enumerate(self.test_dataloader):
                x, y, lengths = data
                bs = len(x)
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # make prediction
                out = self.model(x, lengths)

                # calculate loss
                loss = self.model.loss(out, y)
                batch_loss = loss.item()

                # calculate score (how many largest-indexes match the label value)
                score = (out.argmax(dim=1)==y).sum().item()

                # perform logging and data saving
                batch_accuracy = score/bs*100.0

                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy

                self.model.test_losses.append(batch_loss)
                self.model.test_accs.append(batch_accuracy)

                step = epoch * n_batches + i
                self.writer.add_scalars(f"Loss({self.model.name})", {'valid':batch_loss}, step)
                self.writer.add_scalars(f"Acc({self.model.name})", {'valid':batch_accuracy}, step)

            epoch_loss = epoch_loss / n_batches
            epoch_accuracy = epoch_accuracy / n_batches
            
            print(f"[Valid] Average Loss: {epoch_loss:.6f}")
            print(f"[Valid] Average Acc: {epoch_accuracy:.3f}%")
            
            return (epoch_loss, epoch_accuracy)    



