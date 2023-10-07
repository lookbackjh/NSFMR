import pandas as pd
import numpy as np
from src.data.custompreprocess import CustomOneHot
from src.data.fm_preprocess import FM_Preprocessing
import tqdm
import torch

class Tester:

    def __init__(self, args,model,train_df, test_df, movie_df, user_df) -> None:

        self.args = args
        self.train_df = train_df
        self.test_df = test_df
        self.movie_df = movie_df
        self.user_df = user_df
        self.model=model
        self.original_df=pd.read_csv('dataset/ml-100k/u'+str(args.fold)+'.base',sep='\t',header=None, names=['user_id','movie_id','rating','timestamp'],encoding='latin-1')

    def test_data_generator(self):
        # want to make a dataframe that has user_id, movie_id and for every user_id, movie_id pair 
        movie_ids=sorted(self.movie_df['movie_id'].unique())
        user_ids=sorted(self.original_df['user_id'].unique())

        #dictionary that maps movie_id to movie_frequency
        movie_frequency={}
        for i in movie_ids:
            movie_frequency[i]=len(self.original_df[self.original_df['movie_id']==i])
        
        #dictionary that maps user_id to user_frequency
        user_frequency={}
        for i in user_ids:
            user_frequency[i]=len(self.original_df[self.original_df['user_id']==i])


        # make a dataframe that has all the user_id, movie_id pairs
        npuser_movie=np.zeros((len(user_ids)*len(movie_ids),6))
        npuser_movie=npuser_movie.astype(int)
        npuser_movie[:,0]=np.repeat(user_ids,len(movie_ids))
        npuser_movie[:,1]=np.tile(movie_ids,len(user_ids))
        #2nd column is target
        npuser_movie[:,2]=1
        # 3rd column is c
        npuser_movie[:,3]=1
        # 4th column is user_frequency
        npuser_movie[:,4]=[user_frequency[i] for i in npuser_movie[:,0]]
        # 5th column is movie_frequency
        npuser_movie[:,5]=[movie_frequency[i] for i in npuser_movie[:,1]]

        user_movie=pd.DataFrame(npuser_movie,columns=['user_id','movie_id','target','c','user_frequency','movie_frequency'])

        c=CustomOneHot(user_movie,self.movie_df,self.user_df)
        user_list=user_movie['user_id']
        movie_list=user_movie['movie_id']
        user_movie=c.movieonehot()
        user_movie['user_id']=user_movie['user_id'].astype(int)
        user_movie['movie_id']=user_movie['movie_id'].astype(int)


        movieinfoadded=pd.merge(user_movie,self.movie_df,on='movie_id',how='left')

        userinfoadded=pd.merge(movieinfoadded,self.user_df,on='user_id',how='left')
        
        #drop movie_id
        userinfoadded.drop(['movie_id'],axis=1,inplace=True)

        #pd.getdummies on user_id
        user_df=pd.get_dummies(columns=['user_id'],data=userinfoadded)
        
    

        print(user_df)
        
        return user_df,user_list,movie_list

    
    def test(self):
        
        user_df,user_list,movie_list=self.test_data_generator() 
        #fm=FM_Preprocessing(user_df)
        user_list=user_list.astype(int).unique().tolist()
        movie_list=movie_list.astype(int).unique().tolist()
        self.model.eval()
        for customerid in tqdm.tqdm(user_list[:]):

            cur_customer_id='user_id_'+str(customerid)
            temp=user_df[user_df[cur_customer_id]==1]
            print(temp)
            c_values=temp['c'].values
            y=temp['target'].values
            X=temp.drop(['c','target'],axis=1).values
            X=X.astype(float)
            y=y.astype(float)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)
            c_values_tensor = torch.tensor(c_values, dtype=torch.float32)
            result=self.model.forward(X_tensor)
            topidx=torch.argsort(result,descending=True)[:5]

            
            print("customer id: ",customerid, end=" ")
            print("top 5 recommended product code: ",movie_list[topidx])

        