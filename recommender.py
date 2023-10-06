import torch
import pytorch_lightning as pl
from src.model.deepfm import DeepFM
from src.data.customdata import CustomData
import pandas as pd
from src.data.fm_preprocess import FM_Preprocessing
from src.model.deepfm import DeepFM
from src.model.fm import FactorizationMachine
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

def trainer(args):
    # you can either use your own dataset. 
    train_df = pd.read_pickle('dataset/ml-100k/data_one_hot.pkl')
    train_preprocess = FM_Preprocessing(train_df)
    train_X_tensor=train_preprocess.X_tensor
    train_y_tensor=train_preprocess.y_tensor
    train_c_values_tensor=train_preprocess.c_values_tensor
    train_dataset=CustomData(train_X_tensor,train_y_tensor,train_c_values_tensor)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.model_type=='fm':
        model=FactorizationMachine(train_preprocess.num_features,args.num_factors,args)
    elif args.model_type=='deepfm':
        model=DeepFM(train_preprocess.num_features,args.num_factors,args)

    #model=DeepFM(preprocess.num_features,args.num_factors,args)
    pl.trainer.Trainer(max_epochs=args.num_epochs).fit(model,train_dataloader)
    return model,train_preprocess

if __name__ == '__main__':
    args=parser()
    model,train_preprocess=trainer(args)
