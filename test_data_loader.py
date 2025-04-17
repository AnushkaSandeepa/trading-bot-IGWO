from data_loader import load_daily_data

# Run the data loading function
train_data, test_data = load_daily_data("Data/BTC-Daily.csv")

# Display samples
print("Training Data:")
print(train_data.head())

print("\nTesting Data:")
print(test_data.head())
