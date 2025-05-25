import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("classification_results.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

df["bart_predicted_topic"].value_counts().plot(kind="bar", ax=ax[0], color="skyblue")
ax[0].set_title("Розподіл тем (BART)")
ax[0].tick_params(axis='x', rotation=45)

df["roberta_predicted_topic"].value_counts().plot(kind="bar", ax=ax[1], color="lightgreen")
ax[1].set_title("Розподіл тем (RoBERTa)")
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("models_comparison_plot.png")