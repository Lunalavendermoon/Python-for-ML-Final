'''
Annabel Luo
ACAD 191, Fall 2026
luoannab@usc.edu
Exercise 8
'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas.plotting import scatter_matrix
from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parent
PURCHASE_STYLES = [
    (False, "#0072B2", "No purchase (Revenue=False)"),
    (True, "#D55E00", "Purchase (Revenue=True)"),
]


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
# Side effects: Creates and saves histograms
# Returns: None
def visualize_data(data):
    data.hist(figsize=(14, 12), bins=20)
    plt.suptitle("Online Shoppers - Numerical Attribute Histograms")
    plt.tight_layout()
    plt.savefig(PROJECT_FOLDER / "online_shoppers_histograms.png", dpi=150)


# Function: scatter_plot
# Purpose: Compare product browsing behavior for buyers and non-buyers.
# Parameters: data - the dataset as a DataFrame
# Side effects: Creates and saves a scatter plot
# Returns: None
def scatter_plot(data):
    figure, axis = plt.subplots(figsize=(10, 7))
    for purchased, color, label in PURCHASE_STYLES:
        group = data[data["Revenue"] == purchased]
        axis.scatter(group["ProductRelated"], group["ProductRelated_Duration"],
                     color=color, label=label, alpha=0.35, s=15)
    axis.set_title("Product Browsing Behavior by Purchase Outcome")
    axis.set_xlabel("Number of Product-Related Pages Visited")
    axis.set_ylabel("Total Time on Product-Related Pages")
    axis.legend(title="Purchase outcome", markerscale=2)
    figure.tight_layout()
    figure.savefig(PROJECT_FOLDER / "online_shoppers_scatter.png", dpi=150)


# Function: scatter_all_numeric
# Purpose: Compare every pair of numeric columns, colored by purchase outcome.
# Parameters: data - the dataset as a DataFrame
# Side effects: Creates and saves a scatter plot matrix
# Returns: None
def scatter_all_numeric(data):
    # Draw buyers last so the less common purchase class remains visible.
    plot_data = data.sort_values("Revenue", kind="stable")
    numeric_data = plot_data.select_dtypes(include="number")
    color_map = {purchased: color for purchased, color, label in PURCHASE_STYLES}
    colors = plot_data["Revenue"].map(color_map).to_numpy()
    print("\nSCATTER MATRIX - ALL NUMERIC COLUMNS")
    print(numeric_data.columns.to_list())
    print("Rows plotted:", len(numeric_data))

    axes = scatter_matrix(numeric_data, figsize=(28, 28), alpha=0.3,
                          c=colors, s=4, diagonal="hist",
                          hist_kwds={"bins": 20, "color": "#8c8c8c"})
    for row in axes:
        for axis in row:
            axis.tick_params(axis="both", labelsize=6)
            axis.set_xlabel(axis.get_xlabel(), fontsize=8, rotation=90)
            axis.set_ylabel(axis.get_ylabel(), fontsize=8)
    figure = axes[0, 0].figure
    figure.suptitle("Online Shoppers - All Numeric Columns by Purchase Outcome",
                   fontsize=18, y=0.997)
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for purchased, color, label in PURCHASE_STYLES
    ]
    figure.legend(handles=legend_handles, loc="upper center", ncol=2,
                  fontsize=12, bbox_to_anchor=(0.5, 0.985))
    figure.text(0.5, 0.966, "Diagonal: histograms of all visits combined",
                ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(PROJECT_FOLDER / "online_shoppers_scatter_matrix.png", dpi=150)


# Function: correlations
# Purpose: Calculate Pearson linear correlations for numeric attributes.
# Parameters: data - the dataset as a DataFrame
# Side effects: Prints the correlation matrix and saves it as a CSV
# Returns: None
def correlations(data):
    # Match the example: include every numeric column and Boolean column.
    # OS, browser, region, and traffic codes are nominal categories; their
    # coefficients depend on arbitrary numbering and need careful interpretation.
    corr_matrix = data.corr(numeric_only=True)

    print("\nLINEAR CORRELATIONS (corr())")
    print(corr_matrix.round(3).to_string())
    print("Note: OperatingSystems, Browser, Region, and TrafficType are category codes.")

    print("\nCORRELATIONS WITH PURCHASE OUTCOME (Revenue)")
    print(corr_matrix["Revenue"].drop("Revenue")
          .sort_values(ascending=False).round(3).to_string())

    corr_matrix.to_csv(PROJECT_FOLDER / "online_shoppers_correlations.csv")


# Function: main
# Purpose: Load the CSV and run all three exploratory analysis tasks.
# Parameters: None
# Side effects: Prints dataset information and displays histograms
# Returns: None
def main():
    filename = PROJECT_FOLDER / "online_shoppers_intention.csv"
    print("\nRUNNING SCRIPT:", Path(__file__).resolve())
    print("DATA SOURCE:", filename)
    print("SCATTER COLORS: blue = no purchase; orange = purchase")
    data = load_data(filename)

    if data is not None:
        explore_data(data)
        correlations(data)
        visualize_data(data)
        scatter_plot(data)
        scatter_all_numeric(data)
        print("\nAll three tasks completed. Charts and correlations saved in:")
        print(PROJECT_FOLDER)
        print("Close the plot windows to finish the program.")
        plt.show()


if __name__ == "__main__":
    main()
