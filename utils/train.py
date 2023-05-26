from utils.functions import *


def train_model(data):
    yes_cap = False

    st.markdown("### Basics")

    growth = st.selectbox(
        "**Select Growth Type**",
        ["linear",
        "logistic",
        "flat"]
    )

    cap, hol = st.columns(2)

    with cap:
        yes_cap = st.checkbox('**Add Capacity?**')

        if yes_cap or growth == "logistic":
            yes_cap = True

            col3, col4 = st.columns(2)

            with col3:
                max_log = st.number_input("**Maximum**")

            with col4:
                min_log = st.number_input("**Minimum**", 0.0)

            data['cap'] = max_log

            data['floor'] = min_log

    with hol:
        holidays = None
        yes_holidays = st.checkbox('**Add Holidays?**')
        holidays_prior_scale = 10.0

        if yes_holidays:
            PATH = st.file_uploader('**Upload Holidays Data**')

            if PATH is not None:
                holidays = open_file(PATH)

                if holidays is None:
                    st.error('Cannot open this file.')

                else:
                    holidays['ds'] = pd.to_datetime(holidays['ds'])
                    holidays_prior_scale = st.slider("Holiday Priority Scale", 0.1, 20.8, 10.0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Seasonality")

        seasonality_mode = st.selectbox(
            "**Select Seasonality Mode**",
            [
                "additive",
                "multiplicative"
            ]
        )

        yearly_seasonality = st.selectbox(
            "**Select Yearly Seasonality**",
            [
                "auto",
                True,
                False,
                "Manual"
            ]
        )

        if yearly_seasonality == "Manual":
            yearly_seasonality = st.slider("**Year Seasonality Fourier**", 1, 12, 10)

        weekly_seasonality = st.selectbox(
            "**Select Weekly Seasonality**",
            [
                "auto",
                True,
                False,
                "Manual"
            ]
        )

        if weekly_seasonality == "Manual":
            weekly_seasonality = st.slider("**Weekly Seasonality Fourier**", 1, 52, 3)

        daily_seasonality = st.selectbox(
            "**Select Daily Seasonality** ",
            [
                "auto",
                True,
                False,
                "Manual"
            ]
        )

        if daily_seasonality == "Manual":
            daily_seasonality = st.slider("**Weekly Seasonality Fourier**", 1, 30)

    with col2:
        st.markdown("### Changepoint")

        changepoint_range = st.slider("**Changepoint Range**", 0.1, 1.0, 0.8)
        n_changepoints = st.slider("**Number of Changepoint**", 1, 100, 25)
        changepoint_prior_scale = st.slider("**Changepoint Prior Scale**", 0.01, 0.5, 0.05)
        interval_width = st.slider("**Interval Width**", 0.1, 1.0, 0.8)

    mcmc_samples = st.slider("**MCMC Samples**", 1, 1000, 0)

    # NeuralProphet Object
    with st.spinner(text='Loading Prophet Model'):
        model = Prophet(
            growth=growth,
            holidays=holidays,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_range=changepoint_range,
            n_changepoints=n_changepoints,
            interval_width=interval_width,
            holidays_prior_scale=holidays_prior_scale,
            mcmc_samples=mcmc_samples,
            changepoint_prior_scale=changepoint_prior_scale
        )

    col5, col6 = st.columns(2)

    with col5:
        ADD_YEARLY = st.checkbox('**Add Yearly Fourier?**')

        if ADD_YEARLY:
            year_fourier = st.slider("**Select Number of Yearly Fourier**", 1, 12, 2)
            model.add_seasonality(name='yearly', period=365, fourier_order=year_fourier)

    with col6:
        ADD_MONTHLY = st.checkbox('**Add Monthly Fourier?**')

        if ADD_MONTHLY:
            monthly_fourier = st.slider("**Select Number of Monthly Fourier**", 1, 12, 2)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=monthly_fourier)

    fit = st.button("Fit Prophet")

    if fit:
        with st.spinner(text='Prophet is Fitting'):
            model.fit(data)
            st.success("Model Trained Successfully")

        forecast = model.predict(data)

        with open('./saved_model/prophet_model.json', 'w') as fout:
            fout.write(model_to_json(model))  # Save model

        if yes_cap or growth == 'logistic':
            data = {
                "yes_cap": yes_cap,
                "max": max_log,
                "min": min_log
            }

            with open('./saved_model/model_config.pkl', 'wb') as file:
                pickle.dump(data, file)

        return model, forecast, True

    return None, None, False
