from typing import Any
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn 
# lightning
import pytorch_lightning as pl

class AutoEncoder(pl.LightningModule):

    def __init__(self, input_dim, output_dim, args):
        super(AutoEncoder, self).__init__()
        #here, input can be either user or item
        self.input_dim = input_dim #which is recommended to be one-hot encoded 
        self.output_dim = output_dim #a vector of each user or item
        self.args=args


        self.encoder=nn.Sequential(
            nn.Linear(self.input_dim, self.args.k*2),
            nn.ReLU(),
            nn.Linear(self.args.k*2, self.args.k),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(self.args.k, self.args.k*2),
            nn.ReLU(),
            nn.Linear(self.args.k*2, self.output_dim),
            nn.ReLU()
        )
           

    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

        #loss function
    def loss(self, y_hat, y):
        loss = nn.MSELoss()
        return loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        x,y=batch
        
        y_hat=self.forward(x)
        #batch size * item_number

        loss=self.loss(y_hat, y)
        self.log('train_loss', loss)
        pass
    
    #ADAM optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer
    






