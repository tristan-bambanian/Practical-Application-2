# What Drives the Price of a Car  
[View Jupyter Notebook](What_Drives_the_Price_of_a_Car.ipynb)

### Problem Statement  
Used car dealers are often interested in understanding what factors contribute to and influence a vehicle’s price. In this analysis, I explore a dataset of used cars to identify the characteristics that are most strongly associated with higher or lower prices. I will then present my findings to the used car dealerships and provide recommendations in case they want to adjust their inventory and strategy.

First off, we know our dataset has labels (i.e. prices), so we know that this will be a supervised machine learning problem. As a result, we will either approach this problem using regression or classification techniques. Since price is a continuous variable, we will be using regression techniques to try to predict the price of used cars and better understand the features that have a stronger effect on the price.

### Data Cleaning and EDA
The dataset needed a good amount of cleaning to begin with. I found there were a large amount of duplicate VIN numbers in the dataset (which could be fine if the same car had multiple transactions), but further investigation indicated that the same car was sold at the same price with the only difference being the Region and State columns. I removed these duplicates (and other duplicates) from the dataset and lost confidence in the legitimacy of the Region and State columns.

Univariate analysis of the price column revealed the data was right-skewed (majority of the car prices were low rather than high), so I took a logarithm of the price to achieve a more normal distribution. There were a good amount of rows with price equal to $0, which didn't make logical sense so these rows were removed as well. Moving on to outliers, I used the Z-Score and IQR methods to analyze outliers and found the IQR method to be better for removing outliers from this dataset.

Looking at the correlation of price with the numeric features showed that price had a correlation of 0.39 with year (price tends to increase as year increases) and -0.32 with odometer (price tends to decrease as mileage increases).

Most of the columns in the data were categorical, so I did some bivariate analysis and plotting of these categorical columns against price to get a better idea of which categorical I would keep for the regression modeling. Some categorical columns had too many unique values which would become problemsome for the upcoming categorical encoding since I want to avoid the curse of dimensionality, so I mainly excluded these categorical columns or tried to feature engineer them into less unique values such as the manufacturer column where I grouped the car manufacturers by their geographic regions (America, Europe, Asia). At the end of this bivariate analysis, I decided to only keep the fuel (gas, electric, diesel, etc.) and drive (4wd, fwd, rwd) categorical columns in the dataset.

After all this cleaning there were still some missing values (mainly in the drive column). I didn't have a great way to impute these missing values since the distribution of the drive column was very close between 4wd and fwd, so to be on the safer side, I decided to just drop the missing values rows. Finally, I one hot encoded the fuel and drive categorical columns to make our final dataset numeric. We ultimately went from about 426,000 rows to a little under 77,000 rows.

### Modeling and Interpretation
The first model I used was a standard linear regression model. I split the data into 80% training data and 20% test data. I opted to use mean squared error and root mean squared error for performance evaluation as the root mean squared error is easier for me to interpret since it stays on the same scale as our price data. The linear regression model had a RMSE of about $11,440 so in other words my linear regression model's predictions for the price of a car were about $11,440 off from the actual price on average. The coefficients of the model indicated that:
- Price increases by $684 for every increase in year
- Price wasn't affected by a change in odometer (this was surprising)
- Electric, gas, and hybrid cars were $15,057, $17,621, and $17,039 cheaper respectively than diesel cars on average
- FWD and RWD cars were $10,542 and $579 cheaper than 4WD cars on average

I also tried using Lasso and Ridge regression models to see if they would perform better than my standard linear regression. I first scaled the data, then grid searched the hyperparameter alpha to ensure I used an optimal alpha for each of my Lasso and Ridge models. But, the Lasso and Ridge models didn't appear to do any better than our standard Linear Regression model. Our model score across all 3 models of 46% wasn't that strong, so I probably excluded some important features from the dataset when I did my feature selection on the categorical columns.

### Actionable Items, Next Steps, and Recommendations   
- Sellers should definitely take into account the vehicle's year, mileage, fuel type, and drivetrain as all variables did appear to have an impact on price.
- Dealerships that want to prioritize higher priced inventory should focus on diesel cars, 4wd (four-wheel drive) cars, newer cars, and low mileage cars.
- Since the model's score wasn't that high (46%), there are probably more features in the original dataset that drive the price of the car and warrant further investigation. I would recommend adding back a categorical variable one at a time and testing the models again to see how much they can be improved and what other insights can be derived.
