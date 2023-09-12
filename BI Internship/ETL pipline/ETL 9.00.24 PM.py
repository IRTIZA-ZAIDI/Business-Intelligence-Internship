import pandas as pd
import os
import re
import matplotlib.pyplot as plt


def main():
    try:
        dir_path = "/Users/irtiza/Downloads/Data"
        dict = read_dir(dir_path)
        df = pd.DataFrame(dict)
        # extra line idky
        df = df.drop(0)
        print(df)
        print()

        ############################################################################

        # --REPLACE 0 WITH ZERO
        df1 = df.copy().replace("0", "zero")
        df1.dropna()

        # --NEW COLUMN ADDED
        df1 = df1.assign(
            MergedColumn=df1["Process"].astype(str) + "-" + df1["Branch code"]
        )

        # --REMOVED DUPLICATES
        df1.drop_duplicates(inplace=True)

        print("-" * 150)
        print(df1)
        print()

        ############################################################################

        # -- REJECTED
        df1_rejected_rows = df1.copy()
        # METHOD 1: remove rejected rows where "zero"
        # only include those rows which are not equal to zero
        # df1_rejected_rows = df1_rejected_rows[
        #     df1_rejected_rows["Rows rejected"] != "zero"
        # ]

        # METHOD 2: drop method
        df1_rejected_rows = df1_rejected_rows.drop(
            df1_rejected_rows[df1_rejected_rows["Rows rejected"] == "zero"].index
        )

        print("-" * 150)
        print(df1_rejected_rows)
        print()

        ############################################################################

        # --COMMITTED
        df1_comitted_rows = df1.copy()
        # METHOD 1: only include those rows which are not equal to zero
        # df1_comitted_rows = df1_comitted_rows[
        #     df1_comitted_rows["Rows committed"] != "zero"
        # ]
        # METHOD 2: loops
        # for x in df1_comitted_rows.index:
        #     if df1_comitted_rows.loc[x, "Rows committed"] == "zero":
        #         df1_comitted_rows.drop(x, inplace=True)

        # METHOD 3: drop method
        df1_comitted_rows = df1_comitted_rows.drop(
            df1_comitted_rows[df1_comitted_rows["Rows committed"] == "zero"].index
        )

        print("-" * 150)
        print(df1_comitted_rows)
        print()

        ############################################################################

        # --ROWS > 100
        df1_row_read = df1.copy()
        # Diect filtering in pandas/dont need loops
        df1_row_read = df1_row_read[(df1_row_read["Rows read"] >= 100)]

        print("-" * 150)
        print(df1_row_read)
        print()

        ############################################################################

        # --GROUPING
        # Group branch wise summary of total Rows read and Rows inserted and rejected
        # imp to convert else concating
        df["Rows inserted"] = pd.to_numeric(df["Rows inserted"], errors="coerce")
        df["Rows rejected"] = pd.to_numeric(df["Rows rejected"], errors="coerce")

        # result = df.groupby("Branch code").sum()[
        #     ["Rows read", "Rows inserted", "Rows rejected"]
        # ]

        # better method
        result = df.groupby("Branch code")[
            ["Rows read", "Rows inserted", "Rows rejected"]
        ].sum()
        result = result.rename(
            columns={
                "Rows read": "Total Rows Read",
                "Rows inserted": "Total Rows Inserted",
                "Rows rejected": "Total Rows Rejected",
            }
        )

        print("-" * 150)
        print(result)
        print()

    except FileNotFoundError:
        print("Directory not found")


def read_dir(directory):
    table = [{}]
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            file_path = os.path.join(directory, file)
            ret_values = read_file(file_path)
            code = "BRN-" + str(int(file[:4]))
            process = file[4:].replace(".txt", "")
            new_data = {
                "Process": process,
                "Branch code": code,
                "Rows read": ret_values[0],
                "Rows skipped": ret_values[1],
                "Rows inserted": ret_values[2],
                "Rows updated": ret_values[3],
                "Rows rejected": ret_values[4],
                "Rows committed": ret_values[5],
            }
            table.append(new_data)
    return table


def read_file(file):
    with open(file, "r") as file:
        file_content = file.read()

        rows_read_match = re.search(r"Number of rows read = (\d+)", file_content)
        rows_skipped_match = re.search(r"Number of rows skipped = (\d+)", file_content)
        rows_inserted_match = re.search(
            r"Number of rows inserted = (\d+)", file_content
        )
        rows_updated_match = re.search(r"Number of rows updated = (\d+)", file_content)
        rows_rejected_match = re.search(
            r"Number of rows rejected  = (\d+)", file_content
        )
        rows_committed_match = re.search(
            r"Number of rows committed = (\d+)", file_content
        )

        rows_read = rows_read_match.group(1) if rows_read_match else None
        rows_skipped = rows_skipped_match.group(1) if rows_skipped_match else None
        rows_inserted = rows_inserted_match.group(1) if rows_inserted_match else None
        rows_updated = rows_updated_match.group(1) if rows_updated_match else None
        rows_rejected = rows_rejected_match.group(1) if rows_rejected_match else None
        rows_committed = rows_committed_match.group(1) if rows_committed_match else None

    return [
        int(rows_read),
        rows_skipped,
        rows_inserted,
        rows_updated,
        rows_rejected,
        rows_committed,
    ]


if __name__ == "__main__":
    main()
