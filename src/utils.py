from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=["Con", "Lab"], yticklabels=["Con", "Lab"]
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("BERT Confusion Matrix")
    plt.show()
