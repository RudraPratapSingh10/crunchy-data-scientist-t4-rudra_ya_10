import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "PassengerNumber",
    "GroupSize",
    "CabinNumber",
    "TotalSpending"
]

CATEGORICAL_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side"
]


def create_features(df):
    df = df.copy()

    # PassengerId features
    passenger_split = df["PassengerId"].str.split("_", expand=True)

    df["PassengerNumber"] = passenger_split[1].astype(float)

    group_ids = passenger_split[0]
    group_sizes = group_ids.value_counts()

    df["GroupSize"] = group_ids.map(group_sizes)

    # Cabin features
    cabin_split = df["Cabin"].str.split("/", expand=True)

    df["Deck"] = cabin_split[0]
    df["CabinNumber"] = pd.to_numeric(
        cabin_split[1], errors="coerce"
    )
    df["Side"] = cabin_split[2]

    # Total spending
    spending_columns = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck"
    ]

    df["TotalSpending"] = df[spending_columns].sum(
        axis=1, min_count=1
    )

    # Remove unnecessary columns
    df.drop(
        columns=["PassengerId", "Cabin", "Name", "GroupId"],
        inplace=True,
        errors="ignore"
    )

    return df


def build_preprocessor():

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor