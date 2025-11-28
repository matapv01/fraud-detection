import pandas as pd
from sklearn.model_selection import train_test_split

def data_loader(df, test_size=0.1, random_state=42):
    """
    Chia train/test theo tỷ lệ 90/10.
    Giữ nguyên phân phối Class (0/1) bằng stratify.
    Không cân bằng lại dữ liệu.
    """
    # Shuffle toàn bộ dataset trước
    df_shuffled = df.sample(frac=1, random_state=random_state)

    # Tách X, y
    X = df_shuffled.drop(columns=['Class'])
    y = df_shuffled['Class']

    # Stratified split để giữ tỷ lệ fraud/non-fraud như ban đầu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Gộp lại thành train_df và test_df
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)

    # new_df: train + test
    new_df = pd.concat([train_df, test_df]).sample(frac=1, random_state=random_state)

    # Kiểm tra phân bố
    print("Train distribution:\n", train_df['Class'].value_counts(normalize=True))
    print("Test distribution:\n", test_df['Class'].value_counts(normalize=True))
    print("Original distribution:\n", df['Class'].value_counts(normalize=True))

    return train_df, test_df, new_df
