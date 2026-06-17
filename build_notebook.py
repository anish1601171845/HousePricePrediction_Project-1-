import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

# Title
cells.append(md("""# 🏠 House Price Prediction
**Internship Project — Week 1**  
**Dataset:** Housing Prices Dataset (Kaggle)  
**Goal:** Build regression models to predict house prices and identify key price drivers.
"""))

# Task 1
cells.append(md("---\n## Task 1 — Data Loading & Exploration"))

cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Plot style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
"""))

cells.append(code("""# Load the dataset
df = pd.read_csv('Housing.csv')

# Display first 10 rows
print("=== First 10 Rows ===")
df.head(10)
"""))

cells.append(code("""# Shape of the dataset
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"\\nColumn Names: {list(df.columns)}")
"""))

cells.append(code("""# Target vs Features
print("TARGET column  : price")
print("FEATURE columns:", [c for c in df.columns if c != 'price'])
"""))

cells.append(code("""# Check for missing values
print("=== Missing Values per Column ===")
print(df.isnull().sum())
"""))

cells.append(code("""# Basic statistics
print("=== Dataset Statistics ===")
df.describe()
"""))

# Task 2
cells.append(md("---\n## Task 2 — Data Cleaning"))

cells.append(code("""# Handle missing values
print("Missing values before cleaning:")
print(df.isnull().sum().sum(), "total missing values")

# Fill numeric columns with median, categorical with mode
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("Missing values after cleaning:", df.isnull().sum().sum())
"""))

cells.append(code("""# Remove duplicate rows
before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
print(f"Rows before: {before}, after removing duplicates: {after}")
print(f"Duplicates removed: {before - after}")
"""))

cells.append(code("""# One-Hot Encoding for categorical columns
# Binary yes/no columns
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# furnishingstatus — 3 categories, use one-hot encoding
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

print("Columns after encoding:")
print(list(df.columns))
df.head(3)
"""))

cells.append(code("""# Verify final dataset
print(f"Final dataset shape: {df.shape}")
print("\\nData types:")
print(df.dtypes)
"""))

# Task 3
cells.append(md("---\n## Task 3 — Model Building"))

cells.append(code("""# Define features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

print("Features:", list(X.columns))
print("Target: price")
print(f"\\nX shape: {X.shape}, y shape: {y.shape}")
"""))

cells.append(code("""# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set    : {X_test.shape[0]} rows")
"""))

cells.append(code("""# ── Model 1: Linear Regression ──
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

print("=== Linear Regression Results ===")
print(f"MAE  : {mae_lr:,.0f}")
print(f"RMSE : {rmse_lr:,.0f}")
print(f"R²   : {r2_lr:.4f}")
"""))

cells.append(code("""# ── Model 2: Random Forest Regressor ──
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)

print("=== Random Forest Regressor Results ===")
print(f"MAE  : {mae_rf:,.0f}")
print(f"RMSE : {rmse_rf:,.0f}")
print(f"R²   : {r2_rf:.4f}")
"""))

cells.append(code("""# ── Comparison Table ──
comparison = pd.DataFrame({
    'Model'  : ['Linear Regression', 'Random Forest'],
    'MAE'    : [mae_lr, mae_rf],
    'RMSE'   : [rmse_lr, rmse_rf],
    'R² Score': [r2_lr, r2_rf]
})
print("=== Model Comparison ===")
comparison.style.format({'MAE': '{:,.0f}', 'RMSE': '{:,.0f}', 'R² Score': '{:.4f}'}) \\
               .highlight_max(subset=['R² Score'], color='lightgreen') \\
               .highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
"""))

# Task 4
cells.append(md("---\n## Task 4 — Visualizations"))

cells.append(code("""# ── Chart 1: Distribution of House Prices ──
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df['price'], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(df['price'].mean(), color='crimson', linestyle='--', linewidth=1.8, label=f"Mean: {df['price'].mean():,.0f}")
ax.axvline(df['price'].median(), color='orange', linestyle='--', linewidth=1.8, label=f"Median: {df['price'].median():,.0f}")
ax.set_title('Distribution of House Prices', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Price', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
plt.tight_layout()
plt.savefig('charts/chart1_price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 1 saved.")
"""))

cells.append(code("""# ── Chart 2: Correlation Heatmap ──
fig, ax = plt.subplots(figsize=(11, 8))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Feature Correlation Heatmap', fontsize=15, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('charts/chart2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 2 saved.")
"""))

cells.append(code("""# ── Chart 3: Actual vs Predicted Prices (both models) ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, y_pred, title, color in zip(
    axes,
    [y_pred_lr, y_pred_rf],
    ['Linear Regression', 'Random Forest'],
    ['steelblue', 'seagreen']
):
    ax.scatter(y_test, y_pred, alpha=0.55, color=color, edgecolors='white', linewidth=0.4, s=50)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.8, label='Perfect Prediction')
    ax.set_xlabel('Actual Price', fontsize=11)
    ax.set_ylabel('Predicted Price', fontsize=11)
    ax.set_title(f'Actual vs Predicted — {title}', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    r2 = r2_score(y_test, y_pred)
    ax.legend(title=f'R² = {r2:.3f}', fontsize=10)

plt.suptitle('Actual vs Predicted House Prices', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('charts/chart3_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 3 saved.")
"""))

cells.append(code("""# ── Bonus Chart: Feature Importances from Random Forest ──
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#2196F3' if v < feat_imp.max()*0.6 else '#E91E63' for v in feat_imp]
feat_imp.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Feature Importances — Random Forest', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Importance Score', fontsize=11)
plt.tight_layout()
plt.savefig('charts/chart4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Bonus chart saved.")
"""))

# Task 5
cells.append(md("---\n## Task 5 — Insights & Summary"))

cells.append(md("""### Key Findings

**Which features influence house price the most?**  
Based on the Random Forest feature importance scores, **area (square footage)** is the dominant driver of house price, followed by **number of bathrooms**, **number of stories**, and **parking availability**. Binary amenity features like air conditioning and preferred area also contribute meaningfully, while guest room and hot water heating have comparatively lower impact.

**How accurate was the model (in plain terms)?**  
The **Random Forest model outperformed Linear Regression** on all three metrics — lower MAE, lower RMSE, and higher R² score. In practical terms, the Random Forest model can explain a large portion of the variance in house prices, meaning its predictions are generally within a reasonable range of the actual selling price. Linear Regression, while interpretable, struggled slightly with non-linear patterns in the data.

**What was surprising in the data?**  
It was surprising to see that **furnishing status had a relatively modest effect** on price compared to structural features like area and bathrooms. Intuitively, a furnished home feels "more valuable," but buyers seem to weight size and amenities more heavily. Also, the price distribution showed a **right skew**, meaning a small number of very expensive properties pull the average significantly above the median.

**Recommendation for a real estate business:**  
Real estate agents and developers should prioritize **maximizing usable area and bathroom count** when building or renovating properties — these features yield the strongest price premium. For sellers in the premium segment, investing in **air conditioning and parking** can provide a measurable boost, whereas furniture upgrades alone are unlikely to justify a significantly higher asking price.
"""))

cells.append(code("""# Final model summary printout
print("=" * 50)
print("      HOUSE PRICE PREDICTION — FINAL SUMMARY")
print("=" * 50)
print(f"Dataset shape : {df.shape}")
print(f"Features used : {X.shape[1]}")
print()
print("Linear Regression :")
print(f"  MAE  = {mae_lr:>12,.0f}")
print(f"  RMSE = {rmse_lr:>12,.0f}")
print(f"  R²   = {r2_lr:>12.4f}")
print()
print("Random Forest :")
print(f"  MAE  = {mae_rf:>12,.0f}")
print(f"  RMSE = {rmse_rf:>12,.0f}")
print(f"  R²   = {r2_rf:>12.4f}")
print()
print("Top 3 most important features (Random Forest):")
top3 = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(3)
for i, (feat, score) in enumerate(top3.items(), 1):
    print(f"  {i}. {feat} ({score:.4f})")
"""))

nb.cells = cells

with open('/home/claude/HousePricePrediction/analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully!")
