import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create dataframe
df = pd.read_csv('../scraper/new-pune-shivajinagar-2018-01-01-to-2024-10-05.csv')

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'],format='%d-%m-%y')


# Features: Use 'Normal of Tmax' and 'Normal of Tmin'
X = df[['Normal of Tmax', 'Normal of Tmin']]

# Targets: Predict both 'Tmax' and 'Tmin'
y = df[['Tmax', 'Tmin']]

# Include the 'date' column in X_test for output purposes
dates = df['date']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, dates, test_size=0.3, random_state=42, shuffle=False
)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Round the predictions to 2 decimal places
y_pred = pd.DataFrame(y_pred, columns=['Predicted Tmax', 'Predicted Tmin']).round(2)

# Display predictions alongside actual values and dates
# Exclude 'Normal of Tmax' and 'Normal of Tmin' from the output
results = pd.concat([dates_test.reset_index(drop=True), y_test.reset_index(drop=True), y_pred], axis=1)

# Print the results
print("Predicted Tmax and Tmin for test set with corresponding dates:")
print(results)

# Calculate the Mean Squared Error (MSE) for evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

r2_tmax = r2_score(y_test['Tmax'], y_pred['Predicted Tmax'])
r2_tmin = r2_score(y_test['Tmin'], y_pred['Predicted Tmin'])
print (r2_tmax, r2_tmin)

plt.plot(results['date'],results['Tmax'],label="real")
plt.plot(results['date'],results['Predicted Tmax'],label="pred")
plt.plot(results['date'],results['Tmin'],label="real min")
plt.plot(results['date'],results['Predicted Tmin'],label="pred min")
plt.show()