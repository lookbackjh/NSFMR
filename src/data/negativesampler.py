from copy import deepcopy
import numpy as np
import pandas as pd
import tqdm
class NegativeSampler:
    #  takes input of original dataframe and movie info
    #  make a function that returns negative sampled data

    def __init__(self, original_df, seed) -> None:
        self.original_df = original_df
        self.seed=np.random.seed(seed)
        self.original_df.drop(columns=['timestamp','rating'], axis=1, inplace=True)
        self.original_df['target']=1
        self.original_df['c']=1

        pass

    


    # function taht calculates the c value for each customer and product
    # the higher beta means more weiht on the product frequency, the higher alpha means more weight on the customer frequency
    def get_c(self,df, alpha=.5, beta=.5, gamma=.5, c_zero=.5):
        UF = np.array(df["user_frequency"].astype("float"), dtype=float)
        UF /= np.sum(UF)
        IF = np.array(df["movie_frequency"].astype("float"), dtype=float)
        IF /= np.sum(IF)
        Fs = alpha * beta * IF * UF
        Fs_gamma = Fs ** gamma
        c = Fs_gamma / np.sum(Fs_gamma)
        c = c_zero * c / np.max(c)
        c_appended_df = deepcopy(df)
        c_appended_df['c'] = c

        return c_appended_df
    
    def negativesample(self,checkuniform=False):

        unique_customers = self.original_df['user_id'].unique()
        df=self.original_df
        not_purchased_df = pd.DataFrame()
        ns_df_list = []
        df['user_frequency'] = df.groupby('user_id')['user_id'].transform('count')
        df['movie_frequency'] = df.groupby('movie_id')['movie_id'].transform('count')
        #multiprocess

        print("Negative Sampling Started")

        for customer in tqdm.tqdm(unique_customers):
            unique_products = df['movie_id'].unique()


            customer_frequency = df[df['user_id'] == customer]['user_frequency'].iloc[0]
            purchased_products = df[df['user_id'] == customer]['movie_id'].unique()

            #customer_birth_category = df[df['user_id'] == customer]['BIRTH_YEAR'].iloc[0]
            #customer_gender_category = df[df['user_id'] == customer]['GENDER'].iloc[0]

            not_purchased_df_all = df[~df['movie_id'].isin(purchased_products)]
            not_purchased_codes = not_purchased_df_all['movie_id'].unique()
            negative_sample_products=np.random.choice(not_purchased_codes, int(len(not_purchased_codes) *0.1),replace=False)
            ns_test=df[df['movie_id'].isin(negative_sample_products)][['movie_id','movie_frequency']]
            ns_test=ns_test.drop_duplicates(subset=['movie_id'],keep='first',inplace=False)
            ns_df=pd.DataFrame()

            ns_df['movie_id'] = negative_sample_products
            ns_df=ns_df.assign(user_id=customer)
            #ns_df['AUTH_CUSTOMER_ID'] = customer
            #ns_df=ns_df.assign(BIRTH_YEAR = customer_birth_category)
            #ns_df=ns_df.assign(GENDER= customer_gender_category)
            ns_df=ns_df.assign(user_frequency = customer_frequency)
            ns_df=ns_df.join(ns_test.set_index('movie_id'), on='movie_id')
            # not_purchased_df=pd.concat([not_purchased_df,ns_df],axis=0, ignore_index=True)
            ns_df_list += [ns_df]
        not_purchased_df=pd.concat(objs=ns_df_list, axis=0, ignore_index=True)
        
        not_purchased_df['target'] = 0

        if checkuniform:
            not_purchased_df['c'] = 1
            
        
        else:
            not_purchased_df=self.get_c(not_purchased_df)

        to_return = pd.concat([self.original_df, not_purchased_df], axis=0, ignore_index=True)
        #print(to_return)

        print("Negative Sampling Finished")
        return to_return
    


    


