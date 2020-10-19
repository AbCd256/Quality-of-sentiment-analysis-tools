# Quality-of-sentiment-analysis-tools



# Description 

This project is the material for the paper: Data quality in sentiment analysis tools: The reasons of inconsistency
The project contains implementations of the state-of-the-art sentiment analysis algorithms, the code of the experiment, the results, and information about the datasets used.



# Project-structure

## Code
This directory contains the codes of the project. It is structured as follows:
* `Sentiment_analysis_tools`  containes the six sentiment analysis tools evaluated in the study:
  * Vader:  allows to do sentiment analysis using the word-lexicon vader 
  * senticnet5:  allows to do sentiment analysis using the concept-lexicon Senticnet 
  * sentiwordnet:  allows to do sentiment analysis using the word-lexicon sentiwordnet 
  * rec_nn  allows to do sentiment analysis using the 
  * cnn_text allows to do sentiment analysis using kim's cnn with word embedding from googe news or Gleve
  * char_cnn allows to do sentiment analysis with cnn that use two embedding tpes: word2ve embedding +char embedding
  * bert_cnn: allows to do sentiment analysis using embedding from pretrained bert + cnn 
* `calculate_inconsistency.py` allows to calculate the inconsistency in different sentiment analysis tools
* `hyperparamaters_inc.py`  Provides different tunings possibbilities for sentiment analysis tool on cnn text
* `normalize_dataset.py` Allows to unify the structure of datasets, (unify the polarity, predicted and generalized...) 
* `quality_verification.py` The implimentation of the the heuristics we used to enhance the quality of the generate dataset
 ## Data 
 Data are in the following drive: 
https://drive.google.com/drive/folders/1jdpZtsz06CY6FtYVbvmK_BvrsQc_FIUL?usp=sharing

This directory contains datasets of sentiment analysis extended with paraphrases. each dataset contains a list of analogical sets that where constracted using the generation method 
mentioned in "Adversarial Example Generation with Syntactically Controlled Paraphrase Networks". each dataset has the folowing structure: 
 
 - Id: the Id of the review 
 - Review: The review or the content of the review
 - Golden: Golden standard which is our ground truth. It is the score attributed to the review by human labeling.
 
NB: Reviews that are semantically equivalent have the same Id 


#### Polarity labels 

The different polarity labels in different datasets are attributed as follows: 
* Amazon
     * negative : 0<= Golden <3
	 * neutral : 3
	 * positive : 3<golden <=5 
	 
* Sentiment Treebank 
    * negative : 0 <= golden <= 0.4
   * neutral : 0.4 < golden <= 0.6
   *  positive : 0.6< golden <= 1 
	 
* News headlines dataset
     * negative : -1 <= golden < 0
	 * neutral : golden = 0
	* positive : 0< golden <= 1 

* First GOP debate
 * US airlines tweets
 
 
 NB: 	 
*  Data in the folder "sentiment_dataset_original" represents the original sentiment datasets.
* The folder "sentiment_dataset_with_labels" contains the datasets after converting sentiment score to labels.
*  The folder "sentiment_dataset_augmented" contains the augmented datasets before cleaning 
(data of this folder will e added after acceptation)
* The folder "clean_sentiment_dataset" contains the final clean datasets
(data of this folder will e added after acceptation)

## Experements


This package is organized as follow: 
* `scripts` contains all the scripts of our experements 
* `logs` containes  the logs of our experements
* `plots` containes  different plots of the experements




# Requirements 
This project was developped on Python 3.6.6 linux, RAM 75GB 
```shell
pip install requirement.txt
```
# Program Usage



1- To apply sentiment analysis tools on your data,
```shell
python `method'.py --input_path "input_path".csv  --out_path "out_path".csv
```
  `--input_path`  the path  to the  input data.
 `--out_path`  the path to save the resuts.
 `method` vader, sentiwrdnet, senticnet, rec_nn
2- For machine learning methods: 
* Download a pretrained model from
* Downlod the word2vec pratrained model  
 ### Example
```shell
python text_cnn_predict.py --input_path "input_path".csv   --save_path  "model_path".pl --w2v_path "word2vec_file"  --embedding_type   --out_path "output_path".csv
```
`--input_path` the path  to the  input data.
`--save_path  ` the path to the model 
`--w2v_path ` the path  to the pretrained w2v.
`--embedding_type` the embedding type (Gnews, Glove)
`--out_path` path to save the resulted log
3- You can also  train the CCNs it using your oun data 
```shell
python text_cnn_train.py --input_path  --save_path --w2v_path --embedding_type 
```
4- To reproduce results  : 
* Generate the logs 
  *  for inconsistency experements 
```shell
 python calculate_inconsystency.py --log_path  --out_path  --function 
 ```

`--function` the experement that we awant to generate its logs (intool_inconsistency, intertool_inconsistency, intool_inconsistency_type, intertool_inconsistency_type,  intool_inc_polar_fact, intool_inconsistency_cos_wmd) 
   *  for hyperparamaters tuning experements 
  ```shell
 python hyperparameters_inc.py --input_path --out_path  --function 
 ```
*  for time experements 
```shell
python  time_exp.py --input_path  --out_path
 ```
* Generate the plots
```shell
 python 'exp'.py --log_path  
 ```

`log_path` path to experement logs
exp: the experement we want to run. The experements are in the folder Experements/script




