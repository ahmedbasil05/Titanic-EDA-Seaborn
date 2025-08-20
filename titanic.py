# Project: Titanic Data EDA & Visualization with Seaborn
# Objective: Explore Titanic dataset and show all plots in one figure

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
titanic = sns.load_dataset('titanic')

# Set style
sns.set_style('whitegrid')
sns.set_palette('pastel')
sns.set_context('notebook', font_scale=1)

# Create a figure with multiple subplots
fig, axes = plt.subplots(4, 3, figsize=(20, 18))  # 4 rows, 3 cols
fig.suptitle('Titanic EDA & Visualization', fontsize=22)

# 1. Scatter plot: Fare vs Age by class
sns.scatterplot(data=titanic, x='age', y='fare', hue='pclass', style='sex', ax=axes[0,0])
axes[0,0].set_title('Fare vs Age by Class & Gender')

# 2. Countplot: Passenger count by class & gender
sns.countplot(data=titanic, x='pclass', hue='sex', ax=axes[0,1])
axes[0,1].set_title('Passenger Count by Class & Gender')

# 3. Barplot: Average fare per class
sns.barplot(data=titanic, x='pclass', y='fare', hue='sex', errorbar=None, ax=axes[0,2])
axes[0,2].set_title('Average Fare by Class & Gender')

# 4. Boxplot: Age distribution by class
sns.boxplot(data=titanic, x='pclass', y='age', hue='sex', ax=axes[1,0])
axes[1,0].set_title('Age Distribution by Class & Gender')

# 5. Violin plot: Fare distribution by class
sns.violinplot(data=titanic, x='pclass', y='fare', hue='sex', split=True, ax=axes[1,1])
axes[1,1].set_title('Fare Distribution by Class & Gender')

# 6. Histogram: Age distribution by survival
sns.histplot(data=titanic, x='age', bins=30, kde=True, hue='survived', multiple='stack', ax=axes[1,2])
axes[1,2].set_title('Age Distribution by Survival')

# 7. KDE: Fare distribution by class
sns.kdeplot(data=titanic, x='fare', hue='pclass', fill=True, ax=axes[2,0])
axes[2,0].set_title('Fare Distribution by Class')

# 8. ECDF: Age cumulative distribution by survival
sns.ecdfplot(data=titanic, x='age', hue='survived', ax=axes[2,1])
axes[2,1].set_title('ECDF of Age by Survival')

# 9. Heatmap: Correlation
numeric_cols = titanic.select_dtypes(include='number')
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=axes[2,2])
axes[2,2].set_title('Correlation Heatmap')


# 10. Pairplot can't go in subplot, so show separately
plt.tight_layout()
plt.show()

# Pairplot and jointplot can be shown individually after the main dashboard
sns.pairplot(titanic, vars=['age','fare','sibsp','parch'], hue='survived', palette='Set1', markers=['o','s'])
plt.suptitle('Pairwise Relationships by Survival', y=1.02)
plt.show()

sns.jointplot(data=titanic, x='age', y='fare', hue='survived', kind='reg', palette='coolwarm')
plt.suptitle('Age vs Fare with Regression', y=1.02)
plt.show()
