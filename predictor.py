import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




df= pd.DataFrame()
df_monthly= pd.DataFrame()
supervised_data= pd.DataFrame()


def read_data():
    global df, df_monthly

    df = pd.read_csv('sales_data.csv')
    # Convert the date column to a datetime object
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(['store', 'item'], axis=1)
    # Group the data by month
    df_monthly = df.groupby(pd.Grouper(key='date', freq='M')).sum().reset_index()

    plt.figure(figsize=(15,5))
    plt.plot(df_monthly['date'], df_monthly['sales'])
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Monthly Sales Over 6 Years')
    plt.show()


def sales_diff():
    global df, df_monthly

    df_monthly['sales_difference'] = df_monthly['sales'].diff()
    df_monthly = df_monthly.dropna()
    # Visualisation
    plt.figure(figsize=(15,5))
    plt.bar(df_monthly['date'], df_monthly['sales_difference'], width=12)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Monthly Customer Sales Difference")
    plt.show()


def get_supervised_data():
    global supervised_data

    supervised_data = df_monthly.drop(['date', 'sales'], axis=1)
    for i in range(1,13):
        col_name = 'month_' + str(i)
        supervised_data[col_name] = supervised_data['sales_difference'].shift(i)
    supervised_data = supervised_data.dropna().reset_index()



def split_data():
    global supervised_data

    train_data = supervised_data[:-12]
    test_data = supervised_data[-12:]
    print('Train Data Shape:', train_data.shape)
    print('Test Data Shape:', test_data.shape)



def scale_data():
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    X_train, y_train = train_data[:,1:], train_data[:,0:1]
    X_test, y_test = test_data[:,1:], test_data[:,0:1]
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    print('X_train Shape:', X_train.shape)
    print('y_train Shape:', y_train.shape)
    print('X_test Shape:', X_test.shape)
    print('y_test Shape:', y_test.shape)
    sales_dates = df_monthly['date'][-12:].reset_index(drop=True)
    predict_df = pd.DataFrame(sales_dates)
    act_sales = df_monthly['sales'][-13:].to_list()


def build_model():
    linreg_model = LinearRegression()
    linreg_model.fit(X_train, y_train)
    linreg_pred = linreg_model.predict(X_test)

    linreg_pred = linreg_pred.reshape(-1,1)
    linreg_pred_test_set = np.concatenate([linreg_pred,X_test], axis=1)
    linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)


def model_evaluation():
    result_list = []
    for index in range(0, len(linreg_pred_test_set)):
        result_list.append(linreg_pred_test_set[index][0] + act_sales[index])
    linreg_pred_series = pd.Series(result_list,name='linreg_pred')
    predict_df = predict_df.merge(linreg_pred_series, left_index=True, right_index=True)


    linreg_rmse = np.sqrt(mean_squared_error(predict_df['linreg_pred'], df_monthly['sales'][-12:]))
    linreg_mae = mean_absolute_error(predict_df['linreg_pred'], df_monthly['sales'][-12:])
    linreg_r2 = r2_score(predict_df['linreg_pred'], df_monthly['sales'][-12:])
    print('Linear Regression RMSE: ', linreg_rmse)
    print('Linear Regression MAE: ', linreg_mae)
    print('Linear Regression R2 Score: ', linreg_r2)


def predict():
    plt.figure(figsize=(15,7))
    plt.plot(df_monthly['date'], df_monthly['sales'])
    plt.plot(predict_df['date'], predict_df['linreg_pred'])
    plt.title("Customer Sales Forecast using Linear Regression")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(["Original Sales", "Predicted Sales"])
    plt.show()

