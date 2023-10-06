import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

class DeepFM(pl.LightningModule):
    def __init__(self, num_features, num_factors, args):
        super(DeepFM, self).__init__()
        self.num_features = num_features
        self.num_factors = num_factors
        self.weight_decay = args.weight_decay
        self.lr=args.lr
        
        # FM part
        self.w = nn.Parameter(torch.randn(num_features))
        self.bias=nn.Parameter(torch.randn(1))
        self.v = nn.Parameter(torch.randn(num_features, num_factors))
        
        # Deep part
        input_size = num_features  # Adjust this line to match the shape of your input data
        self.deep_layers = nn.ModuleList()
        for i in range(args.num_deep_layers):
            self.deep_layers.append(nn.Linear(input_size, args.deep_layer_size))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(p=0.2))
            input_size = args.deep_layer_size
        
        self.deep_output_layer = nn.Linear(input_size, 1)
        #self.sig=nn.Sigmoid()
        self.bceloss=nn.BCEWithLogitsLoss()
        
        #self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        #self.loss_func = nn.MSELoss()
    
    def loss(self, y_pred, y_true, c_values):
        #mse = (y_pred - y_true.float()) ** 2
        bce=self.bceloss(y_pred,y_true.float())

        weighted_bce = c_values * bce
        l2_reg = torch.norm(self.w)**2 + torch.norm(self.v)**2  # L2 regularization

        loss_y=torch.mean(weighted_bce) + self.weight_decay * l2_reg
        
        return loss_y

    def forward(self, x):
        # FM part
        linear_terms = torch.matmul(x, self.w)+self.bias
        interactions = 0.5 * torch.sum(
            torch.matmul(x, self.v) ** 2 - torch.matmul(x ** 2, self.v ** 2),
            dim=1,
            keepdim=True
        )
        
        # Deep part
        deep_x = x  # No need to flatten
        for layer in self.deep_layers:
            deep_x = layer(deep_x)
        deep_out = self.deep_output_layer(deep_x)
        #deep_out=self.sig(deep_out)
        y_pred=linear_terms + interactions.squeeze() + deep_out.squeeze()
        
        return y_pred

    def training_step(self, batch, batch_idx):
        x,y,c_values=batch
        y_pred=self.forward(x)
        loss_y=self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) ->Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    
    # def recommend_top_n_items(self, user_features, all_item_features, all_item_ids, top_n=5):
    #     combined_features = torch.cat([user_features.expand(all_item_features.shape[0], -1), all_item_features], dim=1)

    #     with torch.no_grad():
    #         scores = self.forward(combined_features)
    #     # want to normalize scores to be between 0 and 1
    #     scores = (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))
    #     sorted_indices = torch.argsort(scores, descending=True)[:top_n]

    #     return [all_item_ids[i] for i in sorted_indices], scores[sorted_indices]
    
    # def recommend_top_n_items_for_all_users(self, user_features_list, all_item_features, all_item_ids,top_n=5):
    #     # as there are no dataset for testing, we will use all_item_features as user_features_list to recommend for each user
    #     recommendations = {}
    #     scores = {}
    #     for i, user_features in enumerate(user_features_list):
    #         user_id = int(user_features[0].item())  # can replace with actual user ID if I have
    #         top_n_items,score = self.recommend_top_n_items(user_features[1:], all_item_features, all_item_ids, top_n)
    #         recommendations[user_id] = top_n_items
    #         # torch to np array
    #         score=score.cpu().detach().numpy()
    #         scores[user_id] = score
    #     return recommendations, scores
    

    # if you want to save model, uncomment the following function.
    def on_train_end(self) -> None:
        #save model
        torch.save(self.state_dict(), 'model.pth')
        return super().on_train_end()










# class EnsembleFM:
#     def __init__(self, num_features, num_factors, lr=0.01, weight_decay=0.01):
#         self.fm = FactorizationMachine(num_features, num_factors, lr, weight_decay)
#         self.deepfm = FactorizationMachine(num_features, num_factors, lr, weight_decay)
        
#     def fit(self, X, y, num_epochs=10):
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.float32)
        
#         for epoch in range(num_epochs):
#             fm_loss = self.fm.train_step(X_tensor, y_tensor)
#             deepfm_loss = self.deepfm.train_step(X_tensor, y_tensor)
#             print(f'Epoch {epoch+1}/{num_epochs}, FM Loss: {fm_loss:.4f}, DeepFM Loss: {deepfm_loss:.4f}')
    
#     def predict(self, X):
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         with torch.no_grad():
#             fm_pred = self.fm(X_tensor)
#             deepfm_pred = self.deepfm(X_tensor)
#         return fm_pred, deepfm_pred
    
#     def optimize_ensemble_weights(self):
#         def objective(weights):
#             ensemble_pred = weights[0] * fm_pred + weights[1] * deepfm_pred
#             return nn.MSELoss()(ensemble_pred, y_tensor.float())
        
#         fm_pred, deepfm_pred = self.predict(X)
#         y_tensor = torch.tensor(y, dtype=torch.float32)
        
#         res = gp_minimize(objective, [(0.0, 1.0), (0.0, 1.0)], n_calls=20)
#         optimal_weights = res.x
#         return optimal_weights