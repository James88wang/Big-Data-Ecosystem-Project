# Big-Data-Ecosystem-Project

The project allows to focus on an Open Source Distributed Systems:
- First we create a Machine Learning model using Pyspark
- Then we use DVC in order to describe and version control our data sets and models.



## Prerequisites

Before you begin, ensure you have met the following requirement:
  * You have installed the latest version of **PySpark** 
  * You have installed the latest version of **DVC**

From Command Prompt or other recommended consoles:
- Install all packages at once:
`pip install -r src/requirements.txt`
							**OR**
- Install each package manually:
	- To install Pyspark: `pip install pyspark`
	- To install DVC: `pip install dvc`

After installing these prerequisites, we can move on to some more interesting part.

## Data

First of all, we used a data set on card fraud with the idea of designing a model that could predict all fraudulent credit card transactions. The dataset has been provided by Kaggle on the following link: https://www.kaggle.com/mlg-ulb/creditcardfraud

The dataset is composed by 31 columns and more than 284,000 rows and its size about 143 MB. All the columns are numerical values, which doesn't necessitate that much preprocessing work. 

## Models

We have designed 4 models of machine learning in the form of pipelines distributed with pyspark thanks to its library pyspark.ml. The models are the following:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosted 

Then we saved the different models, as well as their scores in order to be able to version each model via DVC. 

## DVC loading

Thanks to DVC, the models can be remotely accessible by running a `dvc pull` after having configured the specific Google Drive folder as the remote storage.

To set the Google Drive folder as your remote storage, follow these simple instructions:

```shell
$ dvc remote add -d storage gdrive://1Y48GIUlQNEmawuT9zWSrNB17lfkFVnf4
```
```shell
$ git commit .dvc/config -m "Configure remote storage"
```
```shell
$ dvc pull
```


## Problems we have encountered

*  More specifically, this was the first time we created our models with Pyspark pipelines and it took us some time to get it right. 

* Trouble in using DVC for versioning our models.



## Contributors

* [@James88wang](https://github.com/James88wang) 💻🐛
* [@HenintsoaRaza](https://github.com/HenintsoaRaza) 💻🐛



## Contact

* james.wang@edu.ece.fr
* henintsoa.razafindrazaka@edu.ece.fr



# Licenses

[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

