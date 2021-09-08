
""" Supervised Learning with scikit-learn

Course Description:
Machine learning is the field that teaches machines and computers to learn from existing data to make predictions 
on new data: Will a tumor be benign or malignant? Which of your customers will take their business elsewhere? 
Is a particular email spam? In this course, you'll learn how to use Python to perform supervised learning, 
an essential component of machine learning. You'll learn how to build predictive models, tune their parameters, 
and determine how well they will perform with unseen data—all while using real world datasets. 
You'll be using scikit-learn, one of the most popular and user-friendly machine learning libraries for Python.
 
 """
""" Chapter 1: Classification
In this chapter, you will be introduced to classification problems and learn how to solve them using supervised learning 
techniques. And you’ll apply what you learn to a political dataset, 
where you classify the party affiliation of United States congressmen based on their voting records.
"""

# Import plotting modules
import matplotlib.pyplot as plt 
import seaborn as sns   
from sklearn.datasets import load_iris

import numpy as np
import pandas as pd

SHOW = True
HIDE = False

class Classification():

    def __init__(self):
        # save load_iris() sklearn dataset to iris
        # if you'd like to check dataset type use: type(load_iris())
        # if you'd like to view list of attributes use: dir(load_iris())
        self.iris = load_iris()

        self.df = pd.DataFrame(data= np.c_[self.iris['data'], self.iris['target'],],
                     columns= self.iris['feature_names'] + ['target'] )

        # iris.target = [0 0 0 ... 1 2]
        # iris.target_name = ['setosa' 'versicolor' 'virginica']
        self.df['species'] = pd.Categorical.from_codes(self.iris.target, self.iris.target_names)

        # print(iris.target, iris.target_names)
        self.prnt(self.df.head(), HIDE)

        self.versicolor_petal_length = self.df.loc[self.df['target'] == 1., 'petal length (cm)']
        self.prnt(self.versicolor_petal_length.head(), HIDE)

        # adding the other species
        self.setosa_petal_length    = self.df.loc[self.df['target'] == 0., 'petal length (cm)']
        self.virginica_petal_length = self.df.loc[self.df['target'] == 2., 'petal length (cm)']




    def prnt(self, data, display = SHOW):
        if display:
            print('###############', '\n', data, '\n')

    def EDA_iris(self, iris):
        X = iris.data
        y = iris.target
        df = pd.DataFrame(X, columns=iris.feature_names)
        self.prnt(df.head())

        # visual EDA
        _ = pd.plotting.scatter_matrix(df, c=y, figsize = [8,8], s=150, marker='D')
        plt.show()

    def KNN(self, df, X_new):
        # Import KNeighborsClassifier from sklearn.neighbors
        from sklearn.neighbors  import KNeighborsClassifier

        # Create arrays for the features and the response variable
        y = df['party'].values
        X = df.drop('party', axis=1).values

        # Create a k-NN classifier with 6 neighbors
        knn = KNeighborsClassifier(n_neighbors = 6)

        # Fit the classifier to the data
        knn.fit(X, y)

        # Predict the labels for the training data X
        y_pred = knn.predict(X)
        # Predict and print the label for the new data point X_new
        new_prediction = knn.predict(X_new)
        print("Prediction: {}".format(new_prediction))






""" Chap 2: Regression

In the previous chapter, you used image and political datasets to predict binary and multiclass outcomes. 
But what if your problem requires a continuous outcome? Regression is best suited to solving such problems. 
You will learn about fundamental concepts in regression and apply them to predict the life expectancy in 
a given country using Gapminder data."""


class Regression(Classification):

    def __init__(self):
        super().__init__()

    def computing_mean(self, versicolor_petal_length):
        # Compute the mean: mean_length_vers
        mean_length_vers = np.mean(versicolor_petal_length)

        # Print the result with some nice formatting
        print('I. versicolor:', mean_length_vers, 'cm')

    
        

""" Chapter 3: Fine-tuning your model

Having trained your model, your next task is to evaluate its performance.
In this chapter, you will learn about some of the other metrics available in scikit-learn 
that will allow you to assess your model's performance in a more nuanced manner. 
Next, learn to optimize your classification and regression models using hyperparameter tuning."""

class Fine_tuning (Regression):
    def __init__(self):
            super().__init__()

   

""" Chapter 4:Preprocessing and pipelines

This chapter introduces pipelines, and how scikit-learn allows for transformers and estimators to be chained together
and used as a single unit. Preprocessing techniques will be introduced as a way to enhance model performance, 
and pipelines will tie together concepts from previous chapters."""

class Preprocessing_pipelines():
    def __init__(self):
        pass



def main():

    chap1 = Classification()  

    # chap1.EDA_iris(chap1.iris)
    
    




if __name__ == "__main__":
    main()