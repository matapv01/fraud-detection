import pandas as pd

def data_loader(df):
    # Shuffle dataset trước
    df_shuffled = df.sample(frac=1, random_state=42)

    # Tách fraud và normal
    fraud_df = df_shuffled[df_shuffled['Class'] == 1]
    non_fraud_df = df_shuffled[df_shuffled['Class'] == 0]

    # --- Test set ---
    n_fraud_test = int(0.1 * len(fraud_df))  # 10% số fraud
    fraud_test = fraud_df.sample(n=n_fraud_test, random_state=42)
    normal_test = non_fraud_df.sample(n=n_fraud_test, random_state=42)  # cân bằng

    test_df = pd.concat([fraud_test, normal_test]).sample(frac=1, random_state=42)

    # --- Train set ---
    fraud_train = fraud_df.drop(fraud_test.index)
    n_fraud_train = len(fraud_train)

    normal_train = non_fraud_df.drop(normal_test.index).sample(n=n_fraud_train, random_state=42)

    train_df = pd.concat([fraud_train, normal_train]).sample(frac=1, random_state=42)

    # --- Gộp train + test thành new_df ---
    new_df = pd.concat([train_df, test_df]).sample(frac=1, random_state=42)

    # Kiểm tra phân bố
    print("Train class distribution:\n", train_df['Class'].value_counts())
    print("Test class distribution:\n", test_df['Class'].value_counts())
    print("New_df class distribution:\n", new_df['Class'].value_counts())

    return train_df, test_df, new_df
