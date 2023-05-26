from utils.libraries import *


def open_file(PATH):
    try:
        data = pd.read_csv(PATH)
        return data
    except:
        print("Couldn't Open CSV File. Trying for EXCEL.")
        try:
            data = pd.read_excel(PATH)
            return data
        except:
            print("Couldn't Open EXCEL File")
        else:
            return None
    else:
        return None


def plot_data(data, original_columns):
    # Showing Data
    plt.figure(figsize=(15, 3))
    plt.title(original_columns[0] + " vs " + original_columns[1], fontsize=20, fontweight="bold")
    plt.plot(data['ds'], data['y'])
    plt.xticks(rotation=-30, ha='right')
    plt.ylabel(original_columns[1])
    plt.xlabel(original_columns[0])
    st.pyplot(plt)


def plot_components(data):
    ses, pred = st.columns(2)

    with ses:
        model = st.selectbox(
            "Type of seasonal component",
            [
                "additive",
                "multiplicative"
            ]
        )

    with pred:
        period = st.slider(
            "Period of the series",
            1,
            math.floor(data.shape[0] / 2),
            math.floor(data.shape[0] / 2)
        )

    data.set_index('ds', inplace=True)
    result = seasonal_decompose(data['y'], period=period, model=model)

    plt.figure(figsize=(15, 3))
    plt.title("Trend", fontsize=20, fontweight="bold")
    result.trend.plot()
    plt.show()
    st.pyplot(plt)
    st.info("**Trend**: Describes whether the time series is decreasing, constant, or increasing over time.")

    plt.figure(figsize=(15, 3))
    plt.title("Seasonality", fontsize=20, fontweight="bold")
    result.seasonal.plot()
    plt.show()
    st.pyplot(plt)
    st.info("**Seasonality**: Describes the periodic signal in your time series.")

    plt.figure(figsize=(15, 3))
    plt.title("Noise", fontsize=20, fontweight="bold")
    result.resid.plot()
    plt.show()
    st.pyplot(plt)
    st.info("**Noise**: The random variations in the time series data.")


def evaluation(y, yhat):
    dict_ = {
        "Mean Absolute Error": [mean_absolute_error(y, yhat)],
        "Mean Absolute Percentage Error": [mean_absolute_percentage_error(y, yhat)],
        "Mean Square Error": [mean_squared_error(y, yhat)],
        "Root Mean Square Error": [math.sqrt(mean_squared_error(y, yhat))]
    }

    metrics = pd.DataFrame(dict_)
    metrics.index = ["Values"]
    return metrics


def forecast_evaluation(eval_data):
    eval_data['Absolute Error'] = abs(eval_data['yhat'] - eval_data['y']) / eval_data['y'] * 100
    error_greater_10 = [err for err in eval_data['Absolute Error'] if err > 18]

    dict_ = {
        "Sum Error": [round(abs(sum(eval_data['yhat']) - sum(eval_data['y'])) / sum(eval_data['y']) * 100, 2)],
        "Max Error": [round(max(eval_data['Absolute Error']), 2)],
        "Min Error": [round(min(eval_data['Absolute Error']), 2)],
        "Count of Error Greater than 10x": [len(error_greater_10)],
        "Percentage of Error Greater than 10%": [round(len(error_greater_10) / len(eval_data) * 100, 2)],
        "Mean Absolute Percentage Error": [round(eval_data['Absolute Error'].mean(), 2)]
    }

    metrics = pd.DataFrame(dict_)
    metrics.index = ["Values"]
    return metrics


def to_monthly_data(data):
    data['Year'] = data['ds'].dt.year
    data['Month'] = data['ds'].dt.month
    data = data.groupby(['Year', 'Month'])['y'].sum().reset_index()
    data['Year'] = data['Year'].astype(str)
    data['Month'] = data['Month'].astype(str)
    data['ds'] = data['Year']+ "-" + data['Month'] + "-" + "1"
    data['ds'] = pd.to_datetime(data['ds'])
    return data[['ds', 'y']]


def data_preprocessing(data):
    # Data Cleaning and Formatting
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds'])
    data.dropna(inplace=True)
    return data
