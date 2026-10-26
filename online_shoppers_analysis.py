'''
Annabel Luo
ACAD 191, Fall 2026
luoannab@usc.edu
Exercise 8
'''

import pandas as pd
import matplotlib.pyplot as plt


# Function: load_data
# Purpose: Read a local CSV and handle possible loading errors.
# Parameters: filename - name of the CSV file
# Side effects: Prints an error message if loading fails
# Returns: DataFrame if successful, None otherwise
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print("Error: CSV file not found:", filename)
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
    except pd.errors.ParserError:
        print("Error: Could not read the CSV format.")
    except OSError as error:
        print("Error opening file:", error)
    return None


# Function: explore_data
# Purpose: Print dataset information, summary statistics, and category counts.
# Parameters: data - the dataset as a DataFrame
# Side effects: Prints information to the console
# Returns: None
def explore_data(data):
    print("\nDATASET SIZE")
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    print("\nDATASET INFORMATION (info())")
    data.info()

    print("\nNUMERICAL SUMMARY (describe())")
    print(data.describe().to_string())

    print("\nVISITOR TYPES (value_counts())")
    print(data["VisitorType"].value_counts().to_string())

    print("\nPURCHASE OUTCOMES (value_counts())")
    print(data["Revenue"].value_counts().to_string())

    print("\nMISSING VALUES")
    print(data.isnull().sum().sum())


# Function: visualize_data
# Purpose: Create histograms for the numerical dataset columns.
# Parameters: data - the dataset as a DataFrame
# Side effects: Displays histograms
# Returns: None
def visualize_data(data):
    data.hist(figsize=(14, 12), bins=20)
    plt.suptitle("Online Shoppers - Numerical Attribute Histograms")
    plt.tight_layout()
    plt.savefig("online_shoppers_histograms.png", dpi=150)
    plt.show()


# Function: main
# Purpose: Load the CSV and run the exploration and visualization functions.
# Parameters: None
# Side effects: Prints dataset information and displays histograms
# Returns: None
def main():
    filename = "online_shoppers_intention.csv"
    data = load_data(filename)

    if data is not None:
        explore_data(data)
        visualize_data(data)


if __name__ == "__main__":
    main()
