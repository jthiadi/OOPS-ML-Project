import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("feature_importances.csv")

top15 = df.head(15)
print(top15)

plt.figure(figsize=(10, 6))

# Scatter plot
plt.scatter(top15["importance"], top15["feature"], color='blue', s=100)  # s is marker size

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances")
plt.gca().invert_yaxis()  # Keep the same order as barh
plt.tight_layout()
plt.savefig("feature_importances_top15_scatter.png")
plt.show()
