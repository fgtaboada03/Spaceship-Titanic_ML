'''Space Titanic Test Cases'''
# test_submission.py
#
# ICS 32
# Project #4: Spaceship Titanic
#
# Machine Learning Pipeline for Kaggle's Spaceship Titanic competition.
#
# NAME: Francisco Taboada
# EMAIL: ftaboada@uci.edu

import unittest
import pandas as pd
from pathlib import Path
from space_titanic import TitanicModel
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split

class Simple_TestCases(unittest.TestCase):
    def test_load_data(self):
        self.knn.load_data()
        self.assertIn(pd.core.frame.DataFrame, self.knn.train_data)
        self.assertIn(pd.core.frame.DataFrame, self.knn.test_data)

class Modeling_TestCases(unittest.TestCase):
    def setUp(self):
        self.knn = TitanicModel(competition_name="spaceship-titanic")
        self.knn.load_data()

    def test_visualize_data(self):
        self.knn.visualize_data(self.knn.train_data)
        assert Path("missing.png").exists()

    def test_visualize_correlation(self):
        self.knn.visualize_correlation(self.knn.train_data)
        assert Path('correlation.png').exists()

    def test_visualize_training_process(self):
        self.knn.encode(self.knn.train_data)
        self.knn.encode(self.knn.test_data)
        assert Path('training_knn.png').exists()

class Modeled_TestCases(unittest.TestCase):
    def setUp(self):
        self.knn = TitanicModel(competition_name="spaceship-titanic")
        self.knn.run()
    
    def test_train_model(self):
        self.assertIn(knn._classification.KNeighborsClassifier ,self.knn.best_model)
        self.assertIn(float, self.knn.best_model)
    
    def test_run(self):
        assert Path('submission.csv').exists()
    
    def test_visualize_training_process(self):
        assert Path('training_knn.png').exists()

class Advanced_Modeled_TestCases(unittest.TestCase):
    def setUp(self):
        SEED = 1234
        self.knn = TitanicModel(competition_name="spaceship-titanic")
        self.encode(self.knn.train_data)
        self.encode(self.knn.test_data)
        self.knn.train_data = self.knn.preprocess(self.knn.train_data)
        self.knn.test_data = self.knn.preprocess(self.knn.test_data)
        self.knn.train_data = self.knn.train_data.dropna(subset=["Transported"])
        X = self.knn.train_data.drop(columns=["Transported"])
        y = self.knn.train_data["Transported"]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.25, random_state=SEED)

        selected_features = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination',
                            'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
                            'VRDeck','Name']
        
        self.X_train_top_features = self.X_train[selected_features]
        self.X_val_top_features = self.X_val[selected_features]
    
    def test_train_model(self):
        self.knn.train_model(self.X_train_top_features, self.y_train, self.X_val_top_features, self.y_val)
        self.assertIn(knn._classification.KNeighborsClassifier ,self.knn.best_model)
        self.assertIn(float, self.knn.best_model)

    def test_run(self):
        assert Path('submission.csv').exists() 
    

if __name__ == '__main__':
    unittest.main()
