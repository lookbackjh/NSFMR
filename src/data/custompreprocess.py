import pandas as pd
import numpy as np
class CustomOneHot:
    # inputs negative sampled data and one hot encodes it
    def __init__(self,args,ns_df,movie_df,user_df) -> None:
        self.args=args
        self.ns_df=ns_df
        self.movie_df=movie_df
        self.user_df=user_df

        pass

    def movieonehot(self):
        
        ncolumns=list(self.ns_df.columns)
        createmoviecolumn=list(self.movie_df['movie_id'].astype(str).unique())
        mcolumn=['movie_'+x for x in createmoviecolumn]
        ncolumns.extend(mcolumn)
        #to_onehot=pd.DataFrame(columns=createmoviecolumn)
        # there are (), i want to remove them
        
        dic_movie = {k: v for v, k in enumerate(ncolumns)}
        dic_ns = {k: v for v, k in enumerate(self.ns_df.columns)}

        
        nsnp=np.array(self.ns_df)
        # create arbritary 2d array with shape (len(fold1), len(createmoviecolumn))
        tmparray=np.zeros((len(nsnp), len(ncolumns)))

        for i,k in enumerate(nsnp):
            # create array for each row
            
            # set user_id
            tmparray[i,dic_movie['user_id']]=k[dic_ns['user_id']]
            # set c
            tmparray[i,dic_movie['c']]=k[dic_ns['c']]
            # set target
            tmparray[i,dic_movie['target']]=k[dic_ns['target']]
            # set movie_id
            tmparray[i,dic_movie['movie_'+str(int(k[dic_ns['movie_id']]))]]=1
            tmparray[i,dic_movie['movie_id']]=k[dic_ns['movie_id']]

            #set user_frequency
            # append to pd_list

        # to dataframe
        temp=pd.DataFrame(tmparray, columns=ncolumns)
        temp.columns = temp.columns.str.replace(r"[\"\',]", '')
        
        #print(temp)

        return temp

    def original_merge(self):
        movieonehot=self.movieonehot()
        # change user_id and movie_id to int
        movieonehot['user_id']=movieonehot['user_id'].astype(int)
        movieonehot['movie_id']=movieonehot['movie_id'].astype(int)
        movieinfoadded=pd.merge(movieonehot,self.movie_df,on='movie_id',how='left')

        userinfoadded=pd.merge(movieinfoadded,self.user_df,on='user_id',how='left')
        
        #drop movie_id
        userinfoadded.drop(['movie_id'],axis=1,inplace=True)

        #pd.getdummies on user_id
        userinfoadded=pd.get_dummies(columns=['user_id'],data=userinfoadded)

        #print(userinfoadded)
        # userinfoadded.drop(['user_frequency'],axis=1,inplace=True)
        # userinfoadded.drop(['movie_frequency'],axis=1,inplace=True)

        return userinfoadded

    def embedding_merge(self,user_embedding,movie_embedding):

        #from trainingdf if user_id is 1, then user_embedding[0] is the embedding
        #from trainingdf if movie_id is 1, then movie_embedding[0] is the embedding

        #user_embedding and movie_embedding are both numpy arrays
        #user_embedding.shape[0] is the number of users
        user_embedding_df=pd.DataFrame()
        movie_embedding_df=pd.DataFrame()

        user_embedding_df['user_id']=sorted(self.ns_df['user_id'].unique())

        movie_embedding_df['movie_id']=sorted(self.ns_df['movie_id'].unique())

        for i in range(user_embedding.shape[1]):
            user_embedding_df['user_embedding_'+str(i)]=user_embedding[:,i]

        for i in range(movie_embedding.shape[1]):
            movie_embedding_df['movie_embedding_'+str(i)]=movie_embedding[:,i]
        
        
        movie_emb_included_df=pd.merge(self.ns_df.set_index('movie_id'), movie_embedding_df,on='movie_id',how='left')
        user_emb_included_df=pd.merge(movie_emb_included_df.set_index('user_id'),user_embedding_df, on='user_id',how='left')


        
        movieinfoadded=pd.merge(user_emb_included_df.set_index('movie_id'),self.movie_df,on='movie_id',how='left')

        userinfoadded=pd.merge(movieinfoadded.set_index('user_id'),self.user_df,on='user_id',how='left')

        # userinfoadded.drop(['user_frequency'],axis=1,inplace=True)
        # userinfoadded.drop(['movie_frequency'],axis=1,inplace=True)

        return userinfoadded






