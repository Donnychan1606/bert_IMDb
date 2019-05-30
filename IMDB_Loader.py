import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv('imdb_master_40_10.csv', encoding='latin-1')
    df = df[df['type'] == 'train']
    df = df[df['review'] != 'train']
    df = df.drop(['id','Unnamed: 0', 'type', 'file'], axis=1)
    df = df[df.label != 'unsup']

    #indexs = list(df[np.isnan(df['label'])].index)
    #df = df.drop(indexs)

    df.label = df['label'].map({'pos': 1, 'neg': 0})
    df.to_csv('./train.csv')
    print(df.shape)
    print('SUCCEED!!!')
    print('END!!')