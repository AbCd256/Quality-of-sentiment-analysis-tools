import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import argparse
#Get files from exp directory

import numpy as np
def intertool_inc(args):
    #sns.set (style="white" , context="talk" , font_scale=0.8 , color_codes=True)
    cmap = sns.diverging_palette (220 , 10 , as_cmap=True)
    mypath= args.log_path
    dirs = [dI for dI in listdir (mypath) if isdir (join (mypath , dI))]
    datasets = ["AMAZON_PRODUCTS" , "NEWS_HEADLINES" , "STANFORD_TREEBANK"]
    tools = ["char_cnn" , "rec_nn" , "senticnet","sentiwordnet"  , "text_cnn" , "vader"]
    fig = plt.figure (figsize=(30,30),facecolor='w')
    #sns.set (style="whitegrid")
    #sns.set (font_scale=0.6)

    #sns.palplot (sns.hls_palette (8 , l=.3 , s=.8))
    numfile = 0
    for dir in dirs:
        path_dir= join(mypath,dir)
        onlyfiles = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
        matrices=[]
        for f in onlyfiles:

            path_file=join(path_dir,f)
            df= pd.read_csv(path_file,sep=";")
            matrix = [
                [round (sum (df['p_p']) / len (df['p_p']) , 3) , round (sum (df['Inc_p_o']) / len (df['Inc_p_o']) , 3) ,
                 round (sum (df['Inc_p_n']) / len (df['Inc_p_n']) , 3)] ,
                [round (sum (df['Inc_p_o']) / len (df['Inc_p_o']) , 3) , round (sum (df['o_o']) / len (df['o_o']) , 3) ,
                 round (sum (df['Inc_o_n']) / len (df['Inc_o_n']) , 3)] ,
                [round (sum (df['Inc_p_n']) / len (df['Inc_p_n']) , 3) ,
                 round (sum (df['Inc_o_n']) / len (df['Inc_o_n']) , 3) , round (sum (df['n_n']) / len (df['n_n']) , 2)]]
            print(matrix)
            matrices.append(matrix)
            print("_________________")
        summatrices= np.add(np.add(np.array(matrices[0]),np.array(matrices[1])), np.array(matrices[2]))/3
        numfile = numfile + 1
        ax = fig.add_subplot (1 , 6 , numfile)

        df_cm = pd.DataFrame (summatrices , ["pos","neut","neg"] ,["pos" , "neut" , "neg"])
        mask = np.zeros_like (df_cm , dtype=np.bool)
        mask[np.triu_indices_from (mask)] = True
        mask[np.diag_indices_from (mask)] = False
        if numfile != 6:

            sns.heatmap (df_cm, mask=mask ,annot=True,cmap=cmap, linewidths=.5,ax=ax, cbar=False)# font size
        else:
            sns.heatmap (df_cm , annot=True ,mask=mask, cmap=cmap , linewidths=.5 , ax=ax )
        ax.set_ylim (3 , 0)
        ax.set_title (dir,fontsize=10)

                #ax.yaxis.set_label_position ("left")


            #ax.tick_params (axis='both' , which='major' , pad=5)
            #ax.set_xlabel('X= Inconsistency degree',fontsize=10)
            #ax.yaxis.set_label_position ("right")
            #df_list.append(df["Inc"])
        #tips = sns.load_dataset(df["Inc"])
    #ax = sns.boxplot(y=df2['Inc'],palette="Set2")
    #tips = sns.load_dataset ("tips")
    #sns.distplot (df_list)
    #g = sns.FacetGrid (tips , col="sex" , hue="smoker")
    #bins = np.linspace (0 , 60 , 13)
    #g.map (plt.hist , "total_bill" , color="steelblue" , bins=bins)
    #plt.subplots_adjust (hspace=0.2,wspace = 0.2)

    plt.show()
    #fig.savefig (args.out_path)

if __name__ == '__main__':
    ##model args
    parser = argparse.ArgumentParser (description='Syntactically Controlled Paraphrase Transformer')

    parser.add_argument ('--log_path' , type=str , default="./data/dataset.csv" ,
                         help='input path of logs')

    parser.add_argument ('--out_path' , type=str , default='./data/sentiment_dataset_sentic.csv' ,
                         help='output of plots')

    args = parser.parse_args ()
    intertool_inc(args)

#gg(gg.mtcars, gg.aes( y=df["Inc"].values)) + gg.geom_boxplot()