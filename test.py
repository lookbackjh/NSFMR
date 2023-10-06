import torch
import pytorch_lightning as pl
from src.model.deepfm import DeepFM
from src.data.customdata import CustomData
import pandas as pd
import numpy as np
# 맘에안듬
import argparse
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_factors', type=int, default=10, help='Number of factors for FM')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=500,    help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=20, help='Number of workers for dataloader')
    parser.add_argument('--num_deep_layers', type=int, default=2, help='Number of deep layers')
    parser.add_argument('--deep_layer_size', type=int, default=128, help='Size of deep layers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='deepfm', help='fm or deepfm')
    parser.add_argument('--topk', type=int, default=10, help='top k items to recommend')


    args = parser.parse_args("")
    return args

args=parser()

x=pd.read_csv('original.csv')
#x.drop(['Unnamed: 0'],axis=1,inplace=True)
# make product dictionary , can als be loaded from json file
productdict={}
for pf in x['movie_id'].unique():
    productdict[pf]=int(x[x['movie_id']==pf]['movie_frequency'].iloc[0])

import tqdm

customer_onehot=pd.get_dummies(x,columns=['user_id'])

customerids=x['user_id'].unique()
productids=x['movie_id'].unique()

# customer id-> dataframe dictionary
customerdict={}

for customerid in tqdm.tqdm(customerids[:2]):
    a=[]
    for pid in productids:
        cur_customer_id='user_id'+str(customerid)
        temp=customer_onehot[customer_onehot[cur_customer_id]==1].iloc[0]
        temp['movie_id']=pid
        temp['movie_frequency']=productdict[pid]
        a.append(temp)
    temp=pd.DataFrame(a)
    temp=pd.get_dummies(temp,columns=['movie_id'])
    customerdict[customerid]=temp

mymodel=DeepFM(7746,10,args)
mymodel.load_state_dict(torch.load("model.pth"))
mymodel.eval()

for customerid in customerids[:2]:
    d=customerdict[customerid]
    c_values=d['c'].values
    y=d['target'].values
    X=d.drop(['c','target'],axis=1).values
    X=X.astype(float)
    y=y.astype(float)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)
    c_values_tensor = torch.tensor(c_values, dtype=torch.float32)
    dataset=CustomData(X_tensor,y_tensor,c_values_tensor)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    result=mymodel.forward(X_tensor)
    top5idx=torch.argsort(result,descending=True)[:5]

    print("customer id: ",customerid, end=" ")
    print("top 5 recommended product code: ",productids[top5idx])