from fileinput import filename
from utils.libraries import *
from utils.functions import *


def future(data):
    st.markdown("### Let's Forecast")

    file_name = st.text_input("**Enter File Name**")
    days = st.slider("**Period?**", 1, 365, 12)

    IS_MONTHLY = st.checkbox("**Is this Monthly Prediction?**")

    if IS_MONTHLY:
        freq = 'MS'
    else:
        freq = 'D'

    tell = st.button("Tell me the Future")

    if tell and file_name != "":
        with open('./saved_model/prophet_model.json', 'r') as fin:
            model = model_from_json(fin.read())  # Load model

        with open('./saved_model/model_config.pkl', 'rb') as file:
            config = pickle.load(file)

        YES_CAP = config["yes_cap"]
        MAX = config["max"]
        MIN = config["min"]

        future = model.make_future_dataframe(periods=days, include_history=False, freq=freq)

        if YES_CAP:
            future['cap'] = MAX
            future['floor'] = MIN

        forecast = model.predict(future)

        fig = model.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), model, forecast)
        st.pyplot(fig)

        st.download_button(
            "Download Forecast",
            forecast[['ds', 'yhat']].to_csv(index=False),
            file_name=file_name + '.csv',
            mime='text/csv'
        )