from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In utils.py
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # Update labels for Stance
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Stance Detection Confusion Matrix")
    # plt.show()
    plt.savefig("plot")
