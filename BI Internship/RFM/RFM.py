import pandas as pd
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def main():
    try:
        transaction_file = "/Users/irtiza/Downloads/bank.xlsx"
        df = read_transactions(transaction_file)
        table = transfrom_table(df)
        rfm = calculate_rfm(table)
        print(rfm)

        print("-" * 100)

        # Statistics
        rfm_stats = rfm[["Recency", "Frequency", "Monetary"]].describe()
        print(rfm_stats)
        print("-" * 100)

        # Correlation
        rfm_corr = rfm[["Recency", "Frequency", "Monetary"]].corr()
        print(rfm_corr)
        print("-" * 100)

        """""
        # Monetary analysis based on mean rank
        low_rfm = rfm[rfm["RFM_Rank"] == "Low Value"]
        mid_rfm = rfm[rfm["RFM_Rank"] == "Mid Value"]
        high_rfm = rfm[rfm["RFM_Rank"] == "High Value"]

        # Create a figure with two rows and two columns of subplots
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        # Plot the bar graph for mean monetary value by RFM rank in the first row, first column
        colors = ["red", "blue", "orange"]
        axs[0, 0].bar(
            ["Low Value", "Mid Value", "High Value"],
            [
                low_rfm["Monetary"].mean(),
                mid_rfm["Monetary"].mean(),
                high_rfm["Monetary"].mean(),
            ],
            color=colors,
        )
        axs[0, 0].set_xlabel("RFM Rank")
        axs[0, 0].set_ylabel("Mean Monetary Value")
        axs[0, 0].set_title("Mean Monetary Value by RFM Rank")

        # Grouped based on RFM_Segment
        segment_counts = rfm.groupby("RFM_Rank")["RFM_Rank"].count()

        # Pie chart
        axs[0, 1].pie(segment_counts, labels=segment_counts.index)
        axs[0, 1].set_title("Piechart RFM Ranks")

        # Histogram
        axs[1, 0].hist(rfm["RFM_Score"], bins=5, edgecolor="black")
        axs[1, 0].set_xlabel("RFM Score")
        axs[1, 0].set_ylabel("Frequency")
        axs[1, 0].set_title("Histogram of RFM Scores")

        # Remove the empty subplot in the second row, second column
        axs[1, 1].axis("off")

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()
        """

        # Figure with two rows and three columns of subplots
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))

        # Monetary analysis based on mean rank
        low_rfm = rfm[rfm["RFM_Rank"] == "Low Value"]
        mid_rfm = rfm[rfm["RFM_Rank"] == "Mid Value"]
        high_rfm = rfm[rfm["RFM_Rank"] == "High Value"]

        # Mean monetary value bargraph
        colors = ["red", "blue", "orange"]
        axs[0, 0].bar(
            ["Low Value", "Mid Value", "High Value"],
            [
                low_rfm["Monetary"].mean(),
                mid_rfm["Monetary"].mean(),
                high_rfm["Monetary"].mean(),
            ],
            color=colors,
        )
        axs[0, 0].set_xlabel("RFM Rank")
        axs[0, 0].set_ylabel("Mean Monetary Value" + "\n" + "Unit dollars:$")
        axs[0, 0].set_title("Mean Monetary Value by RFM Rank")

        # First row, second column
        axs[0, 1].bar(
            ["Low Value", "Mid Value", "High Value"],
            [
                low_rfm["Recency"].mean(),
                mid_rfm["Recency"].mean(),
                high_rfm["Recency"].mean(),
            ],
            color=colors,
        )
        axs[0, 1].set_xlabel("RFM Rank")
        axs[0, 1].set_ylabel("Mean Recency Value")
        axs[0, 1].set_title("Mean Recency Value by RFM Rank")

        # Mean frequency value bargraph
        axs[0, 2].bar(
            ["Low Value", "Mid Value", "High Value"],
            [
                low_rfm["Frequency"].mean(),
                mid_rfm["Frequency"].mean(),
                high_rfm["Frequency"].mean(),
            ],
            color=colors,
        )
        axs[0, 2].set_xlabel("RFM Rank")
        axs[0, 2].set_ylabel("Mean Frequency Value")
        axs[0, 2].set_title("Mean Frequency Value by RFM Rank")

        # Grouped based on RFM_Segment
        segment_counts = rfm.groupby("RFM_Rank")["RFM_Rank"].count()

        # Pie chart
        axs[1, 0].pie(segment_counts, labels=segment_counts.index)
        axs[1, 0].set_title("Piechart RFM Ranks")

        # Histogram
        axs[1, 1].hist(rfm["RFM_Score"], bins=5, edgecolor="black")
        axs[1, 1].set_xlabel("RFM Score")
        axs[1, 1].set_ylabel("Frequency")
        axs[1, 1].set_title("Histogram of RFM Scores")

        # Remove the empty subplot in the second row, third column
        axs[1, 2].axis("off")

        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()

    except FileNotFoundError or FileExistsError:
        print("Error in file location")


# df = pd.DataFrame(dict)


def read_transactions(file):
    # Read the Excel file into a DataFrame
    try:
        df = pd.read_excel(file)
        df = data_cleaning(df)
        return df
    except FileNotFoundError or FileExistsError:
        print("Error in file location")


def data_cleaning(df):
    # filled null with 0
    df = df.fillna(0)

    # Removed duplicates
    df = df.drop_duplicates()

    # Corrected formats and type
    df["Account No"] = df["Account No"].str.strip()
    df["Account No"] = df["Account No"].str.replace("'", "")
    df["Account No"] = df["Account No"].astype(int)
    df["WITHDRAWAL AMT"] = df["WITHDRAWAL AMT"].astype(int)
    df["DEPOSIT AMT"] = df["DEPOSIT AMT"].astype(int)

    # Dropped unnessary columns
    df = df.drop(["CHQ.NO.", "TRANSACTION DETAILS", "BALANCE AMT", "."], axis=1)

    return df


def transfrom_table(df):
    reference_date = df[
        "DATE"
    ].max()  # Use the latest transaction date as the reference date
    df["Recency"] = (reference_date - df["DATE"]).dt.days
    df["Transaction Amount"] = df["WITHDRAWAL AMT"].astype(int) + df[
        "DEPOSIT AMT"
    ].astype(int)

    frequency = df.groupby("Account No")["DATE"].count().reset_index()
    frequency.columns = ["Account No", "Frequency"]

    monetary = df.groupby("Account No")["Transaction Amount"].sum().reset_index()
    monetary.columns = ["Account No", "Monetary"]

    recency = df.groupby("Account No")["Recency"].mean().reset_index()
    recency.columns = ["Account No", "Recency"]

    rfm = pd.merge(
        pd.merge(frequency, monetary, on="Account No"), recency, on="Account No"
    )

    return rfm


def calculate_rfm(rfm):
    rfm["Monetary"] = rfm["Monetary"] / 1000000

    # Calculate Recency,Frequency,and Monetary Score
    rfm["Recency_Score"] = 10 - pd.qcut(rfm["Recency"], q=10, labels=False)
    rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"], q=10, labels=False)
    rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], q=10, labels=False)

    # Calculate RFM Score
    rfm["RFM_Score"] = (
        rfm["Recency_Score"] * 2 / 10
        + rfm["Frequency_Score"] * 3 / 10
        + rfm["Monetary_Score"] * 5 / 10
    ) / 2

    conditions = [
        rfm["RFM_Score"] >= 5,
        rfm["RFM_Score"] >= 3.5,
        rfm["RFM_Score"] >= 2.5,
        rfm["RFM_Score"] >= 1,
    ]
    choices = ["Best Value", "High Value", "Mid Value", "Low Value"]

    rfm["RFM_Rank"] = np.select(conditions, choices, default="Low Value")

    return rfm


if __name__ == "__main__":
    main()
