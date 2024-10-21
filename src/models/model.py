import os

import torch
import torch.nn as nn

import numpy as np

# base class
class Model(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.save_dir = self.init_save_dir()

        self.layer_stack = None
        self.criterion = None
        self.optim = None

        # logs
        self.train_losses = []
        self.test_losses = []

    def init(self, lr):
        self.layer_stack = self.init_layer_stack()
        self.criterion = self.init_criterion()
        self.optim = self.init_optim()  
        self.set_learning_rate(lr)

    def init_layer_stack(self):
        raise NotImplementedError("Required implementation for init_layer_stack.")

    def init_criterion(self):
        raise NotImplementedError("Required implementation for init_criterion.")

    def init_optim(self):
        raise NotImplementedError("Required implementation for init_optim.")

    def init_save_dir(self):
        return os.path.join(os.path.dirname(__file__), "checkpoints/" + self.name) if self.name is not None else None

    def save_model(self):
        if self.name == "": 
            print("Error: must provide a name to save model checkpoint.")
            return       
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

        # save layer architecture
        np.save(os.path.join(self.save_dir, self.name + "_model_layers.npy"), np.array(self.layer_stack))

        # save weights
        torch.save(self.state_dict(), os.path.join(self.save_dir, self.name + ".pth"))

        # save logs
        self.save_logs()
        print("Saved model checkpoint to:", self.save_dir)

    def save_logs(self):
        if self.name == "": 
            print("Error: must provide a name to save model logs.")
            return     
            
        np.save(os.path.join(self.save_dir, self.name + "_train_losses.npy"), np.array(self.train_losses))
        np.save(os.path.join(self.save_dir, self.name + "_test_losses.npy"), np.array(self.test_losses))

    def load_model(self):
        if self.name == "": 
            print("Error: must provide a name to load existing model checkpoint.")
            return False
        if not os.path.exists(os.path.join(self.save_dir, self.name + ".pth")): 
            print("Error: unable to load model checkpoint.")
            return False
            
        self.load_state_dict(torch.load(os.path.join(self.save_dir, self.name + ".pth")))
    
        # now load logs
        self.load_logs()
        print("Loaded model checkpoint.")
        return True

    def load_logs(self):
        if self.name == "": 
            print("Error: must provide a name to load existing model logs.")
            return
        if not os.path.exists(os.path.join(self.save_dir, self.name + ".pth")): return
        if not os.path.exists(os.path.join(self.save_dir, self.name + "_train_losses.npy")): return

        self.train_losses = np.load(os.path.join(self.save_dir, self.name + "_train_losses.npy")).tolist()
        self.test_losses = np.load(os.path.join(self.save_dir, self.name + "_test_losses.npy")).tolist()
    
    def __repr__(self):
        return str(self.layer_stack)

    def forward(self, x):
        return self.layer_stack(x)

    def loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def set_learning_rate(self, lr):
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def clear_logs(self):
        self.logs = []
