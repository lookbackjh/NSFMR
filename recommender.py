import torch
import pytorch_lightning as pl
from src.model.deepfm import DeepFM
from data.customdataloader import CustomDataLoader
import pandas as pd
from src.data.fm_preprocess import FM_Preprocessing
from src.model.deepfm import DeepFM
from src.model.fm import FactorizationMachine
import argparse
from src.data.negativesampler import NegativeSampler
from src.data.custompreprocess import CustomOneHot
from src.data.movielensdata import MovielensData
from src.tester import Tester

def parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_factors', type=int, default=15, help='Number of factors for FM')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=120,    help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=20, help='Number of workers for dataloader')
    parser.add_argument('--num_deep_layers', type=int, default=2, help='Number of deep layers')
    parser.add_argument('--deep_layer_size', type=int, default=128, help='Size of deep layers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='fm', help='fm or deepfm')
    parser.add_argument('--topk', type=int, default=5, help='top k items to recommend')
    parser.add_argument('--fold', type=int, default=1, help='fold number')
    args = parser.parse_args("")
    return args



def getdata(fold):

    movielens=MovielensData('dataset/ml-100k','u.data',fold=fold)
    train,test=movielens.data_getter()
    movie_info=movielens.movie_getter()
    user_info=movielens.user_getter()

    ## 1. Negative Sampling
    ns=NegativeSampler(train,seed=42)
    nssampled=ns.negativesample(isuniform=False)

    ## 2. one hot encoding

    onehot=CustomOneHot(nssampled,movie_info,user_info)
    train=onehot.infomerge()

    return train, test,movie_info,user_info



def trainer(args,train_df):
    # trainer for each fold
    
    #print(train_df)
    train_preprocess = FM_Preprocessing(train_df)
    train_X_tensor=train_preprocess.X_tensor
    train_y_tensor=train_preprocess.y_tensor
    train_c_values_tensor=train_preprocess.c_values_tensor
    
    train_dataset=CustomDataLoader(train_X_tensor,train_y_tensor,train_c_values_tensor)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    if args.model_type=='fm':
        model=FactorizationMachine(train_preprocess.num_features,args.num_factors,args)
    elif args.model_type=='deepfm':
        model=DeepFM(train_preprocess.num_features,args.num_factors,args)

    #model=DeepFM(preprocess.num_features,args.num_factors,args)
    pl.trainer.Trainer(max_epochs=args.num_epochs).fit(model,train_dataloader)
    return model,train_df



if __name__ == '__main__':
    args=parser()
    folds=[1,2,3,4,5]
    results=[]
    for i in range(1,6):
        args.fold=i
        train_df ,test,movie_info,user_info=getdata(args.fold)
        print("fold ",i," data loaded")
        model,train_preprocess,test=trainer(args,train_df)
        tester=Tester(args,model,train_df,test,movie_info,user_info)
        result=tester.test()
        results.append(result)
    
    print("fold 1 to 5 results")
    for i in range(5):
        print("fold ",i+1," result:",end=" ")
        print(results[i])


