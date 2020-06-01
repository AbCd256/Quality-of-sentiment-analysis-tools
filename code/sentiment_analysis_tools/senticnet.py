import argparse
import numpy as np
import gensim
import pandas as pd
from sentic import SenticPhrase
from os.path import isfile , join
import operator


#---------------------// sentiment analysis with sentic //------------------------------------------#
def sentiment_analyzer_scores(sentence):

    sp1 = SenticPhrase (sentence)

    score = sp1.get_sentiment (sentence)

    if "positive" in score: score = 'Positive'

    if "negative" in score: score = 'Negative'

    if "neutral" in score: score = 'Neutral'

    return score

if __name__ == '__main__':

    ##model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--input_path' , type=str , default="./data/dataset.csv" ,
                         help='parse load path')

    parser.add_argument ('--out_path' , type=str , default='./data/sentiment_dataset_sentic.csv' ,
                         help='data save path')

    args = parser.parse_args()


    #input dataframes
    df_input=pd.read_csv(args.input_path,sep=";", encoding = "ISO-8859-1")
    #df_input=df_input.fillna(value=pd.np.nan, inplace=True)
    #df_input.columns=["Id","Review","Golden","Pred",'Dist']

    #df_input['Dist']=df_input['Dist'].replace(np.nan, 0,inplace=True)
    #output dataframes
    df_output=pd.DataFrame (columns=["Id","Review","Golden","Pred"])


    for i in range (len (df_input)):

        sentimentData_sentence1 = sentiment_analyzer_scores (df_input.iloc[i]["Review"])

        df_output.loc[i] = [df_input.iloc[i]["Id"] , df_input.iloc[i]["Review"] , df_input.iloc[i]["Golden"] ,
                             sentimentData_sentence1]

    df_output.to_csv (args.out_path, sep=";", index=False)



