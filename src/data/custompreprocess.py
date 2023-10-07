import pandas as pd
import numpy as np
class CustomOneHot:
    # inputs negative sampled data and one hot encodes it
    def __init__(self,ns_df,movie_df,user_df) -> None:
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
            tmparray[i,dic_movie['user_frequency']]=k[dic_ns['user_frequency']]
            #set movie_frequency
            tmparray[i,dic_movie['movie_frequency']]=k[dic_ns['movie_frequency']]
            # append to pd_list

        # to dataframe
        temp=pd.DataFrame(tmparray, columns=ncolumns)
        temp.columns = temp.columns.str.replace(r"[\"\',]", '')
        
        #print(temp)

        return temp

    def infomerge(self):
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
        userinfoadded.drop(['user_frequency'],axis=1,inplace=True)
        userinfoadded.drop(['movie_frequency'],axis=1,inplace=True)

        return userinfoadded

