import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Sample data
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1])
y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])

classes = list(set(y_true))
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# Plotting the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True)

# Rotate x-axis labels by 45 degrees and set font size
plt.xticks(rotation=45, fontsize=6)

# Rotate y-axis labels by 45 degrees and set font size
plt.yticks(rotation=45, fontsize=6)

plt.title("Confusion Matrix")
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")

plt.show()
