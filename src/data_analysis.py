# =============================================================
# Credit Card Fraud Detection - Data Analysis Script (.py version)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------
# Config
# -----------------------------
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

# -----------------------------
# Load dataset
# -----------------------------

def data_analysis(df, idx):
    print("\n=== Loading dataset ===")

    print("Shape:", df.shape)

    # -----------------------------
    # Basic info
    # -----------------------------
    print("\n=== Data Info ===")
    print(df.info())

    print("\n=== Describe (transposed) ===")
    print(tabulate(df.describe().T, headers='keys', tablefmt='psql'))

    # -----------------------------
    # Samples
    # -----------------------------
    print("\n=== First 5 rows ===")
    print(tabulate(df.head(), headers='keys', tablefmt='psql'))

    print("\n=== Last 5 rows ===")
    print(tabulate(df.tail(), headers='keys', tablefmt='psql'))

    print("\n=== 5 Fraud Samples ===")
    print(tabulate(df[df['Class'] == 1].sample(5, random_state=42), headers='keys', tablefmt='psql'))

    print("\n=== 5 Normal Samples ===")
    print(tabulate(df[df['Class'] == 0].sample(5, random_state=42), headers='keys', tablefmt='psql'))

    # -----------------------------
    # Missing values & duplicates
    # -----------------------------
    print("\n=== Missing values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\n=== Duplicate rows ===")
    print("Duplicates:", df.duplicated().sum())

    # -----------------------------
    # Class distribution
    # -----------------------------
    print("\n=== Class Distribution ===")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print("Fraud Ratio:", class_counts[1] / class_counts.sum())

    plt.figure(figsize=(5,4))
    sns.countplot(x='Class', data=df)
    plt.title("Class Distribution")
    plt.savefig(f"visualize/class_distribution{idx}.png")

    # -----------------------------
    # Amount distribution
    # -----------------------------
    print("\n=== Plot Amount Distribution ===")
    plt.figure(figsize=(8,4))
    sns.histplot(df['Amount'], bins=100, log_scale=(False, True))
    plt.title("Amount Distribution (Highly Skewed)")
    plt.savefig(f"visualize/amount_distribution{idx}.png")

    df["logAmount"] = np.log1p(df["Amount"])

    plt.figure(figsize=(8,4))
    sns.histplot(df['logAmount'], bins=50)
    plt.title("Log(Amount + 1) Distribution")
    plt.savefig(f"visualize/log_amount_distribution{idx}.png")

    # -----------------------------
    # Hour extraction
    # -----------------------------
    df["Hour"] = (df["Time"] // 3600) % 24

    plt.figure(figsize=(10,4))
    sns.countplot(x='Hour', data=df)
    plt.title("Transactions per Hour")
    plt.savefig(f"visualize/transactions_per_hour{idx}.png")

    # -----------------------------
    # Amount vs Class
    # -----------------------------
    plt.figure(figsize=(8,4))
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.yscale("symlog")
    plt.title("Amount by Class (symlog scale)")
    plt.savefig(f"visualize/amount_by_class{idx}.png")

    plt.figure(figsize=(8,4))
    sns.violinplot(x='Class', y='logAmount', data=df)
    plt.title("Log Amount by Class")
    plt.savefig(f"visualize/log_amount_by_class{idx}.png")

    # -----------------------------
    # Correlation matrix
    # -----------------------------
    print("\n=== Correlation with Class ===")
    corr = df.corr()
    print(corr['Class'].sort_values(ascending=False))

    plt.figure(figsize=(18,12))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.savefig(f"visualize/correlation_matrix{idx}.png")

    # -----------------------------
    # PCA Visualization
    # -----------------------------
    print("\n=== PCA Visualization ===")
    cols = [c for c in df.columns if c not in ["Class", "Time", "Hour"]]
    X_scaled = StandardScaler().fit_transform(df[cols])
    y = df['Class'].values

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[y==0,0], X_pca[y==0,1], s=2, alpha=0.3, label="Normal")
    plt.scatter(X_pca[y==1,0], X_pca[y==1,1], s=8, alpha=0.8, label="Fraud")
    plt.legend()
    plt.title("PCA 2D Projection")
    plt.savefig(f"visualize/pca_projection{idx}.png")


    print("\n=== DONE: All plots saved ===")



import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def distribution_plot(new_df, idx=None, save_path=None):
    """
    Vẽ distribution plot cho các cột V14, V12, V10 của giao dịch gian lận.
    
    Parameters:
    - new_df: DataFrame chứa dữ liệu
    - idx: không sử dụng trong hàm, giữ nguyên cho tương thích
    - save_path: đường dẫn file để lưu hình ảnh (ví dụ: 'output.png')
    """
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # V14
    v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
    sns.histplot(v14_fraud_dist, ax=ax1, kde=True, color='#FB8861')
    ax1.set_title('V14 Distribution \n(Fraud Transactions)', fontsize=14)

    # V12
    v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
    sns.histplot(v12_fraud_dist, ax=ax2, kde=True, color='#56F9BB')
    ax2.set_title('V12 Distribution \n(Fraud Transactions)', fontsize=14)

    # V10
    v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
    sns.histplot(v10_fraud_dist, ax=ax3, kde=True, color='#C5B3F9')
    ax3.set_title('V10 Distribution \n(Fraud Transactions)', fontsize=14)

    plt.tight_layout()

    # Lưu hình
    plt.savefig(f"visualize/feature_distributions{idx}.png")


import matplotlib.pyplot as plt
import seaborn as sns

def boxplot_outliers(new_df, features, save_name):
    """
    Vẽ boxplot cho các feature chỉ ra outliers và lưu ảnh.
    
    Parameters:
    - new_df: DataFrame chứa dữ liệu
    - features: list các feature cần vẽ (ví dụ: ['V14','V12','V10'])
    - save_name: tên file để lưu ảnh (ví dụ: 'boxplot.png')
    """
    num_features = len(features)
    f, axes = plt.subplots(1, num_features, figsize=(6*num_features, 6))

    if num_features == 1:
        axes = [axes]  # đảm bảo axes là list nếu chỉ có 1 feature

    colors = ['#B3F9C5', '#f9c5b3']

    for i, feature in enumerate(features):
        sns.boxplot(x="Class", y=feature, data=new_df, ax=axes[i], palette=colors)
        axes[i].set_title(f"{feature} Feature \n Reduction of outliers", fontsize=14)
        # annotation, chỉnh xy theo giá trị trung bình của feature
        y_min = new_df[feature].min()
        axes[i].annotate('Fewer extreme \n outliers', xy=(0.95, y_min), xytext=(0, y_min*0.6),
                         arrowprops=dict(facecolor='black'),
                         fontsize=12)

    plt.tight_layout()
    plt.savefig(f"visualize/boxplot_{save_name}")