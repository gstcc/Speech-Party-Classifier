from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from transformers import Trainer


class WeightedTrainer(Trainer):
    def __init__(self, *args, weights_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.class_weights = weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def plot_confusion_matrix(y_true, y_pred, labels=None, mode=None):
    cm = confusion_matrix(y_true, y_pred)    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",           
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto"
    )
    
    plt.ylabel("Actual Category")
    plt.xlabel("Predicted Category")
    plt.title("Confusion Matrix")
    
    if labels is not None:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
    plt.tight_layout() 
    plt.savefig(f"plot_{mode}.png")
    plt.close()
