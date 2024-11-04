import os

import torch
import torch.nn as nn

import numpy as np

# base class
class Model(nn.Module):
    # def __init__(self, name, hyper_params):
    def __init__(self, name):
        super().__init__()
        self.name = name
        # self.hyper_params = hyper_params
        self.save_dir = self.init_save_dir()

        self.layer_stack = None
        self.criterion = None
        self.optim = None

        # logs
        self.train_losses = []
        self.train_accs = []
        self.valid_losses = []
        self.valid_accs = []

    def init(self, lr):
        self.layer_stack = self.init_layer_stack()
        self.criterion = self.init_criterion()
        self.optim = self.init_optim()  
        self.set_learning_rate(lr)

    def get_name_with_hyper_params(self) -> str:
        # return f"{self.name}_{self.hyper_params['bs']}_{self.hyper_params['lr']}_{self.hyper_params['epochs']}"
        return f"{self.name}"

    def init_layer_stack(self):
        raise NotImplementedError("Required implementation for init_layer_stack.")

    def init_criterion(self):
        raise NotImplementedError("Required implementation for init_criterion.")

    def init_optim(self):
        raise NotImplementedError("Required implementation for init_optim.")

    def init_save_dir(self):
        if self.name is None: return None
        ret = os.path.join("./checkpoints/" + self.name) 
        # os.makedirs(ret)
        return ret

    def save_model(self):
        if self.name == "": 
            print("Error: must provide a name to save model checkpoint.")
            return       
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

        # save layer architecture
        if self.layer_stack is not None:
            np.save(os.path.join(self.save_dir, "model_layers.npy"), np.array(self.layer_stack))

        # save weights
        torch.save(self.state_dict(), os.path.join(self.save_dir,"model.pth"))

        # save logs
        self.save_logs()
        # print("Saved model checkpoint to:", self.save_dir)

    def save_logs(self):
        if self.name == "": 
            print("Error: must provide a name to save model logs.")
            return     
            
        np.save(os.path.join(self.save_dir, "train_losses.npy"), np.array(self.train_losses))
        np.save(os.path.join(self.save_dir, "train_accs.npy"), np.array(self.train_accs))
        np.save(os.path.join(self.save_dir, "valid_losses.npy"), np.array(self.valid_losses))
        np.save(os.path.join(self.save_dir, "valid_accs.npy"), np.array(self.valid_accs))

    def load_model(self):
        if self.name == "": 
            print("Error: must provide a name to load existing model checkpoint.")
            return False
        if not os.path.exists(os.path.join(self.save_dir, "model.pth")): 
            print("Error: unable to load model checkpoint.")
            return False
            
        self.load_state_dict(torch.load(os.path.join(self.save_dir, "model.pth"), weights_only=True))
    
        # now load logs
        self.load_logs()
        print("Loaded model checkpoint.")
        return True

    def load_logs(self):
        if self.name == "": 
            print("Error: must provide a name to load existing model logs.")
            return
        if not os.path.exists(os.path.join(self.save_dir, "model.pth")): return
        if not os.path.exists(os.path.join(self.save_dir, "train_losses.npy")): return

        self.train_losses = np.load(os.path.join(self.save_dir, "train_losses.npy")).tolist()
        self.train_accs = np.load(os.path.join(self.save_dir, "train_accs.npy")).tolist()
        self.valid_losses = np.load(os.path.join(self.save_dir, "valid_losses.npy")).tolist()
        self.valid_accs = np.load(os.path.join(self.save_dir, "valid_accs.npy")).tolist()
    
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
