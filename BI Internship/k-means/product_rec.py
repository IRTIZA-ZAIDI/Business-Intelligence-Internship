import pandas as pd
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.utils import parallel_backend


sns.set(
    context="notebook",
    palette="Spectral",
    style="darkgrid",
    font_scale=1.5,
    color_codes=True,
)


def main():
    try:
        transaction_file = "/Users/irtiza/Downloads/Model.csv"
        df = read_data(transaction_file)
        correlation_matrix = df.corr()
        print(correlation_matrix)
        # df = scaling(df)
        apply_kmean(df)
        # apply_new(df)

    except FileNotFoundError or FileExistsError:
        print("Error in file location")


def read_data(file):
    # Read the Excel file into a DataFrame
    try:
        # df = pd.read_csv(file, index_col="customer_id")
        df = pd.read_csv(file)
        df.head()
        df = data_cleaning(df)
        print(df)

        return df
    except FileNotFoundError or FileExistsError:
        print("Error in file location")


def data_cleaning(df):
    df["job_industry_category"].fillna("Others", inplace=True)

    df.dropna(subset=["product_id"], inplace=True)
    df.drop("name", axis=1, inplace=True)
    df.drop("total_online_order", axis=1, inplace=True)

    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    df["gender"] = df["gender"].astype(int)

    label_encoder = LabelEncoder()
    df["job_industry_category_encoded"] = label_encoder.fit_transform(
        df["job_industry_category"]
    )
    df.drop("job_industry_category", axis=1, inplace=True)

    df["owns_car"] = df["owns_car"].map({"Yes": 1, "No": 0})

    df["tenure"] = df["tenure"].astype(int)

    return df


def scaling(df):
    # Select the columns to be scaled
    columns_to_scale = [
        "total_transaction",
        "average_transaction_amount",
        "tenure",
        "past_3_years_bike_related_purchases",
    ]

    scaler = StandardScaler()
    # df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    print(df)

    return df


from sklearn.model_selection import GridSearchCV


def apply_kmean(df):
    data_array = np.array(df)
    k_values = range(1, 10)
    wcss = []

    # Iterate over each value of k
    for k in k_values:
        # Fit the k-means model to the data
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(data_array)

        # Calculate the sum of squared distances for the current k
        wcss.append(kmeans.inertia_)

    # Elbow curve
    plt.plot(k_values, wcss, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Curve")
    plt.show()

    # Assuming you have a DataFrame called 'df' with the specified columns
    columns = [
        "transaction_count",
        # "total_transaction",
        "average_transaction_amount",
        # "past_3_years_bike_related_purchases",
        "age",
        # "product_id",
        "job_industry_category_encoded",
        "owns_car",
        "tenure",
    ]

    X = df[columns].values

    # Optimal number of clusters
    n_clusters = 3

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)

    # Add the cluster labels to the DataFrame
    df["Cluster"] = y_kmeans

    # Visualize the clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["product_id"],
        df["average_transaction_amount"],
        c=df["Cluster"],
        cmap="viridis",
    )
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.set_title("K-Means Clustering")
    ax.set_xlabel("Product ID")
    ax.set_ylabel("Average Transaction Amount")
    plt.show()

    predicted_labels = kmeans.labels_

    print("////////////////////")
    silhouette_avg = silhouette_score(X, predicted_labels)
    print("Silhouette Score: ", silhouette_avg)


"""''
def apply_new(df):
    columns = [
        "transaction_count",
        "average_transaction_amount",
        "age",
        "job_industry_category_encoded",
        "owns_car",
        "tenure",
    ]

    X = df[columns].values

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        "n_clusters": [2, 3, 4, 5],
        "init": ["k-means++", "random"],
        "n_init": [10, 20, 30],
    }

    # K-means model
    kmeans = KMeans(random_state=42)

    # Create a silhouette scorer
    silhouette_scorer = make_scorer(silhouette_score)

    # Create an instance of GridSearchCV
    grid_search = GridSearchCV(kmeans, param_grid, scoring=silhouette_scorer)

    # Perform hyperparameter tuning with parallel processing
    with parallel_backend("threading"):
        grid_search.fit(X)

    # Get the best hyperparameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Fit the best model to the data
    best_model.fit(X)

    # Predict the cluster labels
    y_pred = best_model.predict(X)

    # Add the cluster labels to the DataFrame
    df["Cluster"] = y_pred

    # Visualize the clusters
    plt.scatter(
        df["product_id"],
        df["average_transaction_amount"],
        c=df["Cluster"],
        cmap="viridis",
    )
    plt.xlabel("Product ID")
    plt.ylabel("Average Transaction Amount")
    plt.title("K-Means Clustering")
    plt.show()

    # Calculate the silhouette score
    silhouette_avg = silhouette_score(X, y_pred)
    print("Silhouette Score: ", silhouette_avg)

"""
if __name__ == "__main__":
    main()
