from typing import Any
import torch
import torch.nn as nn
#lightning
import pytorch_lightning as pl

class FactorizationMachine(pl.LightningModule):
    def __init__(self, num_features, num_factors, args):
        super(FactorizationMachine, self).__init__()
        self.num_features = num_features
        self.num_factors = num_factors
        self.w = nn.Parameter(torch.randn(num_features))
        self.bias=nn.Parameter(torch.randn(1))
        self.v = nn.Parameter(torch.randn(num_features, num_factors))
        #self.loss_func = nn.BCEWithLogitsLoss()
        #self.loss_func =  nn.MSELoss()
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.bceloss=nn.BCEWithLogitsLoss()
    

    def loss(self, y_pred, y_true,c_values):
        # calculate weighted mse with l2 regularization
        #mse = (y_pred - y_true.float()) ** 2
        bce = self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * bce
        l2_reg = torch.norm(self.w)**2 + torch.norm(self.v)**2
        return torch.mean(weighted_bce) + self.weight_decay * l2_reg
    
    def forward(self, x):
        # FM part loss with interaction terms
        linear_terms = torch.matmul(x, self.w)+self.bias
        interactions = 0.5 * torch.sum(
            torch.matmul(x, self.v) ** 2 - torch.matmul(x ** 2, self.v ** 2),
            dim=1,
            keepdim=True
        )

        return linear_terms + interactions.squeeze()
    
    def training_step(self, batch, batch_idx):
        x,y,c_values=batch
        y_pred=self.forward(x)
        loss_y=self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) ->Any:

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

    





    
