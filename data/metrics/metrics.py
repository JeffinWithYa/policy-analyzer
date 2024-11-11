import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
def loadJson(filePath):
    with open(filePath, "r") as f:
        return json.load(f)
    
def normalize(jsonData):
    # Initialize lists to store match values for metric calculation
    trueSlice = []
    predictedSlice = []

    # Extract the relevant fields for comparison
    for entry in jsonData:
        modelCategory = list(entry.get("model_analysis", {}).get("category", {}).keys())
        if modelCategory:
            modelKey = modelCategory[0]
        else:
            continue

        # get all top level human annotations
        humanAnnotations = list({list(annotation.keys())[0] for annotation in entry.get("human_annotations", [])})

        if modelKey in humanAnnotations:
            predictedSlice.append(modelKey)
            trueSlice.append(modelKey)
        else:
            predictedSlice.append(modelKey)
            trueSlice.append("NO_MATCH")

    return trueSlice, predictedSlice

def normalizeSubcategoryMatch(jsonData):
    # Subcategory match works by looking at the model prediction subcategories, and if any one of the subcategory predictions
    # matches one of the annotators subcategory labelling then we have a match. It only has to match one of them, not all.

    # Initialize lists to store match values for metric calculation
    trueSlice = []
    predictedSlice = []

    # Extract the relevant fields for comparison
    for entry in jsonData:
        modelCategory = entry.get("model_analysis", {}).get("category", {})
        if not modelCategory:
            continue
        
        modelKey = list(modelCategory.keys())[0]
        modelSubCategories = modelCategory[modelKey]

        # get all top level human annotations
        humanAnnotations = entry.get("human_annotations", [])

        foundMatch = False
        for annotation in humanAnnotations:
            # First check if top level category matches
            topCategory = list(annotation.keys())[0]
            if topCategory == modelKey:
                # check if any subcategory key-value pair matches
                humanSubCategory = annotation[topCategory]
                for k, v in modelSubCategories.items():
                    if humanSubCategory.get(k) == v:
                        foundMatch = True
                        break # found a match no need to keep checking subcategories
            
            if foundMatch:
                break # found a match, no need to keep checking human annotations

        if foundMatch:
            predictedSlice.append(modelKey)
            trueSlice.append(modelKey)
        else:
            predictedSlice.append(modelKey)
            trueSlice.append("NO_MATCH")

    return trueSlice, predictedSlice

def calcConfusionMatrix(trueSlice, predictedSlice):
    # Calculate Metrics, using weighted because of imbalance in frequency of categories.
    # Weighted averaging makes sure that common classes like "Other" doesn"t overly influence results, 
    # and rare classes don"t disproportionaly skew the results. 
    # Using macro would be interesting.. but would give disproportionate advantage to the common class and disadvantage the rare classes
    # Using micro would give the gross performance globally and may also be interesting (total TP/FP/etc. across each instance unweighted)
    # zero_devision is needed because the predicted class never has a value of "NO_MATCH which would lead ti devisions by zero when there
    # are no true positives or false positives for that class"
    accuracy = accuracy_score(trueSlice, predictedSlice)
    precision = precision_score(trueSlice, predictedSlice, average="weighted", zero_division=0)
    recall = recall_score(trueSlice, predictedSlice, average="weighted", zero_division=0)
    f1 = f1_score(trueSlice, predictedSlice, average="weighted", zero_division=0)

    return accuracy, precision, recall, f1

def getBarGraph(data, outFile):
    barW = 0.10
    numMetrics = len(data[0][1])
    numPlots = len(data)
    plt.figure(figsize=(26,6))

    barPositions = [np.arange(numMetrics) + barW * i for i in range(numPlots)]
    colours = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for i, (label, values) in enumerate(data):
        plt.bar(barPositions[i], values, color=colours[i % len(colours)], width=barW, edgecolor="grey", label=label)

    plt.title("Bar Graph of LLM Performance in Annotating Privacy Policies", fontweight="bold", fontsize=15)
    plt.ylabel("Performance Metric Value", fontweight="bold")
    plt.xticks(barPositions[0] + barW * (numPlots - 1) / 2, ["Accuracy", "Precision", "Recall", "F1 Score"])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig("data/metrics/output/" + outFile)
    plt.close()

def getHeatMap(data, outFile):
    # unpacks the data list into individual tuples, and then passes to zip function to create tuple of labels and tuple of metrics data
    labels, metrics = zip(*data)
    dataFrame = pd.DataFrame(metrics, index=labels, columns=["Accuracy", "Precision", "Recall", "F1 Score"])

    plt.figure(figsize=(15,6))
    axes = sns.heatmap(dataFrame, annot=True, cmap="coolwarm_r", cbar=True, fmt=".2f")
    colourbar = axes.collections[0].colorbar
    colourbar.set_label("Performance Metric Value", rotation=270, labelpad=20)
    plt.title("Heatmap of LLM Performance in Annotating Privacy Policies", fontweight="bold", fontsize=15)
    plt.xlabel("Metrics")
    plt.ylabel("LLMs")
    plt.savefig("data/metrics/output/" + outFile)
    plt.close()
    
def main():
    # Get all files from directory.
    directory = "data/"
    plotData = []
    subCategoryPlotData = []

    for filename in os.listdir(directory):
        if filename.startswith("analysis_results_") and filename.endswith(".json"):
            label = filename.replace("analysis_results_", "").replace(".json", "")

            filePath = os.path.join(directory, filename)
            data = loadJson(filePath)
            actual, predicted = normalize(data)
            subCategoriesIncludedActual, subCategoriesIncludedPredicted = normalizeSubcategoryMatch(data)

            accuracy, precision, recall, f1 = calcConfusionMatrix(actual, predicted)
            subAccuracy, subPrecision, subRecall, subF1 = calcConfusionMatrix(subCategoriesIncludedActual, subCategoriesIncludedPredicted)


            plotData.append((label, [accuracy, precision, recall, f1]))
            subCategoryPlotData.append((label, [subAccuracy, subPrecision, subRecall, subF1]))

    getBarGraph(plotData, "TopLevelCategories_Bargraph.png")
    getHeatMap(plotData, "TopLevelCategories_Heatmap.png")
    getBarGraph(subCategoryPlotData, "SubCategories_Bargraph.png")
    getHeatMap(subCategoryPlotData, "SubCategories_Heatmap.png")

if __name__ == "__main__":
    main()