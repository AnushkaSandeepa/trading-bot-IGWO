import pandas as pd

def load_daily_data(file_path="data/BTC-USD-Daily.csv"):
    
    # Load the CSV file
    df = pd.read_csv("Data/BTC-Daily.csv")
    df
    # Rename columns if needed (in case headers differ)
    df.columns = [col.strip().lower() for col in df.columns]

    # Parse date column
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date just in case
    df = df.sort_values(by='date')

    # Drop rows with missing close prices
    df = df.dropna(subset=['close'])

    # Filter relevant columns: Date, Open, High, Low, Close, Volume
    df = df[['date', 'open', 'high', 'low', 'close']]

    # Split into train (before 2020) and test (2020 onward)
    train_df = df[df['date'] < '2020-01-01'].reset_index(drop=True)
    test_df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)

    return train_df, test_df
