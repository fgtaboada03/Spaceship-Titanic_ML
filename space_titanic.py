'''Space Titantic ML Algorithm'''
# space_titanic.py
#
# ICS 32
# Project #4: Spaceship Titanic
#
# Machine Learning Pipeline for Kaggle's Spaceship Titanic competition.
#
# NAME: Francisco Taboada
# EMAIL: ftaboada@uci.edu
# STUDENT ID: 56522406

from pathlib import Path
import time
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from tqdm import tqdm

SEED = 1234

class TitanicModel:
    '''ML Model'''
    def __init__(self, competition_name: str, download_path: str = "data", file_name: str = "submission.csv"):
        self.competition_name = competition_name
        self.download_path = Path(download_path)
        self.file_name = file_name
        self.api = KaggleApi()
        self.api.authenticate()
        self.train_data = None
        self.test_data = None
        self.best_model = None
        self.best_score = 0

    # Data Acquisition
    def download_dataset(self) -> None:
        """Downloads and extracts Kaggle dataset."""
        self.download_path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset: {self.competition_name}...")
        self.api.competition_download_files(self.competition_name, self.download_path)

        zip_path = self.download_path / f"{self.competition_name}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.download_path)
            zip_path.unlink()
        else:
            print(f"Error: {zip_path} not found.")

    def load_data(self) -> None:
        """
        Loads training and test datasets.
        If not, call download_dataset() to download data from Kaggle.
        Read train.csv and store returned data frame in self.train_data.
        Read test.csv and store returned data frame in self.test_data.
        """
        if not self.download_path.exists():
            self.download_dataset()
        path = Path(self.download_path)
        _, test_data, train_data = Path.iterdir(path)

        with open(train_data, "r") as data:
            self.train_data = pd.read_csv(data)
            print(type(self.train_data))
        with open(test_data, "r") as data:
            self.test_data = pd.read_csv(data)

    # Data Preprocessing

    @staticmethod
    def preprocess(data: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values in the dataset with the median of each column."""
        data_copy = data.copy()

        most_frequent_input = SimpleImputer(strategy="most_frequent")
        median_input = SimpleImputer(strategy='median')
        mean_input = SimpleImputer(strategy='mean')

        # before encoding 0.5795768169273229

        mean = ['RoomService', 'FoodCourt']

        median = ['Spa','Age','VRDeck', 'ShoppingMall', 'FoodCourt']

        mode = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP',
       'Name', 'ShoppingMall']

        # cryosleep == True or age <= 12, spending must be 0.0
        data_copy.loc[data_copy['CryoSleep'] == True] = data_copy.loc[data_copy['CryoSleep'] == True].fillna(0.0)

        # replace earth planets no VIP
        data_copy.loc[(data_copy['HomePlanet'] == 'Earth') & (data_copy['VIP'].isna()), 'VIP'] = False

        # fill missing names
        data_copy[['Name']] = data_copy[['Name']].fillna('Missing')

        # unify group info
        # Create a new column for group ID by extracting the first portion of PassengerId
        data_copy['GroupID'] = data_copy['PassengerId'].str.split('-').str[0]

        # Columns to fill (modify as needed)
        fill_cols = ['Destination', 'HomePlanet']

        # Fill NaN values within each group
        data_copy[fill_cols] = data_copy.groupby('GroupID')[fill_cols].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))


        # mean
        for category in mean:
            data_copy[[category]] = mean_input.fit_transform(data_copy[[category]])

        # median
        for category in median:
            data_copy[[category]] = median_input.fit_transform(data_copy[[category]])

        # mode
        for category in mode:
            data_copy[[category]] = most_frequent_input.fit_transform(data_copy[[category]])

        return data_copy


    # Model Training
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Trains and selects the best KNN model using hyperparameter tuning."""
        training_scores = {}

        if X_train.isnull().sum().sum() > 0 or X_val.isnull().sum().sum() > 0:
            X_train = self.preprocess(X_train)
            X_val = self.preprocess(X_val)

        # Iterate over different values of n_neighbors (k)
        for n_neighbors in tqdm(range(1, 51), desc="Tuning k-NN Hyperparameters"):
            # Initialize KNN
            model = knn(n_neighbors=n_neighbors)

            # Train KNN model
            model.fit(X_train, y_train)

            # Calculate accuracy on validation data
            y_pred = model.predict(X_val)

            score = accuracy_score(y_val, y_pred)

            training_scores[n_neighbors] = score  # Store the accuracy value directly

            # Track the best model based on accuracy
            if score > self.best_score:
                self.best_model = model
                self.best_score = score

        print(score)
        print(f"Best model {self.best_model} achieved validation accuracy: {self.best_score}")
        self.visualize_training_process(training_scores)


    # Visualization
    @staticmethod
    def visualize_data(data: pd.DataFrame) -> None:
        """Visualizes missing data and feature correlations."""
        # Visualize missing data
        print(f"\n   Sample of data: \n\n {data.head()}")
        print(f"\n   Shape of data: {data.shape}")
        data.info()
        print(data.describe())

        missing = data.isnull().sum()
        columns = list(missing.index)
        counts = list(missing.values)

        # Save the figure in missing.png
        # with columns as x values and counts as y values
        # use the following code to avoid overlapping x-axis labels
        # plt.xticks(rotation=45)
        # start a new figure with "plt.figure(figsize=(12, 8))"

        plt.figure(figsize=(12,6), layout='constrained')
        plt.plot(columns, counts)
        plt.xticks(rotation=45)
        plt.savefig('missing.png')

    @staticmethod
    def visualize_correlation(data: pd.DataFrame) -> None:
        '''Visualizes correlation'''
        # Calculate correlation with 'Transported'
        correlation = data.corrwith(data['Transported'])
        correlation = correlation.drop('Transported').sort_values(ascending=False)
        columns = list(correlation.index)
        counts = list(correlation.values)

        # Save in correlation.png
        # Use plt.xticks(rotation=) to avoid overlap
        # X-axis label is Features, and y-axis label is Correlation

        plt.figure(figsize=(10,8), layout='constrained')
        plt.plot(columns, counts)
        plt.xlabel("Features")
        plt.ylabel("Correlation")
        plt.xticks(rotation=45)
        plt.savefig('correlation.png')

    @staticmethod
    def visualize_training_process(training_scores: dict[int, float]) -> None:
        """Visualizes training and validation accuracy during hyperparameter tuning for k-NN."""
        # Plot accuracy for each value of n_neighbors
        # Save plot in training_knn.png"
        # training_scores is a dictionary where
        # key = n_neighbors and value = score

        plt.figure(figsize=(10,6), layout='constrained')
        plt.plot(training_scores.keys(), training_scores.values())
        plt.savefig('training_knn.png')

    @staticmethod
    def encode(data: pd.DataFrame) -> None:
        '''encodes columns PassengerId to str'''
        # Have to clean the data and encode strings into numbers
        for col in data.select_dtypes(include="object"):
            if col != "PassengerId": # problem with passenger id column
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # Submission
    def create_submission_file(self, predictions: pd.Series, passenger_ids: pd.Series) -> None:
        """Saves predictions in Kaggle submission format."""
        submission = pd.DataFrame({
            "PassengerId": passenger_ids,
            "Transported": predictions
        })
        submission.to_csv(self.file_name, index=False)
        print(f"Submission saved to {self.file_name}")

    def submit(self, competition_name: str, file_name: str) -> None:
        '''submits results'''
        self.api.competition_submit(file_name, message="Final Predictions", competition=competition_name)
        print("Submission successful!")

        time.sleep(10)
        submissions = self.api.competition_submissions(competition_name)

        for idx, submission in enumerate(submissions):
            print(f"Submission ID: {submission.ref}")
            print(f"Public Score: {submission.public_score}")
            print("-" * 40)
            if idx == 3:
                break

        return submissions

    # Main Pipeline
    def run(self) -> None:
        """Executes the entire ML pipeline."""
        # Step 1: Data Acquisition
        self.load_data()

        print("Step 1")

        # Step 2: Visualize Missing Data
        self.visualize_data(self.train_data)

        print("Step 2")

        # Step 3: Encode Data

        self.train_data = self.preprocess(self.train_data)
        self.test_data = self.preprocess(self.test_data)
        
        self.encode(self.train_data)
        self.encode(self.test_data)

        print("Step 3")

        # Step 4: Visualize Correlation
        self.visualize_correlation(self.train_data)

        print("Step 4")

        # Step 5: Data Preprocessing

        # self.train_data = self.preprocess(self.train_data)
        # self.test_data = self.preprocess(self.test_data)

        print("Step 5")

        # Drop rows with missing values in 'Transported'
        self.train_data = self.train_data.dropna(subset=["Transported"])

        # Prepare Data for Training
        X = self.train_data.drop(columns=["Transported"])
        y = self.train_data["Transported"]

        # Split dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=SEED)

        # Step 6: Baseline KNN Model Training - uses all features
        # This baseline KNN model should achieve validation accuracy of ~58%

        # baseline
        # selected_features = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination',
        #                     'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
        #                     'VRDeck','Name']

        # before encoding my best
        selected_features = ['HomePlanet', 'CryoSleep','Destination','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        # after encoding my best
        # selected_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa','VRDeck']

        # for i in range(2**13):
        #     selector = {
        #         'PassengerId': (i >> 0) & 1,
        #         'HomePlanet': (i >> 1) & 1,
        #         'CryoSleep': (i >> 2) & 1,
        #         'Cabin': (i >> 3) & 1,
        #         'Destination': (i >> 4) & 1,
        #         'Age': (i >> 5) & 1,
        #         'VIP': (i >> 6) & 1,
        #         'RoomService': (i >> 7) & 1,
        #         'FoodCourt': (i >> 8) & 1,
        #         'ShoppingMall': (i >> 9) & 1,
        #         'Spa': (i >> 10) & 1,
        #         'VRDeck': (i >> 11) & 1,
        #         'Name': (i >> 12) & 1
        #     }

        #     selected = [key for key, value in selector.items() if value]

        #     if not selected:
        #         continue

        X_train_top_features = X_train[selected_features]
        X_val_top_features = X_val[selected_features]

        self.train_model(X_train_top_features, y_train, X_val_top_features, y_val)

        # Step 8: Prediction
        passenger_ids = self.test_data["PassengerId"]
        self.test_data = self.test_data[X_train_top_features.columns]
        predictions = self.best_model.predict(self.test_data)

        # Step 9: Submission

        self.create_submission_file(predictions, passenger_ids)
        self.submit("spaceship-titanic", "submission.csv")

if __name__ == "__main__":
    titanic_model = TitanicModel(competition_name="spaceship-titanic")
    titanic_model.run()
