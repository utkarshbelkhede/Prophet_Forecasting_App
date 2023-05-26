from utils.libraries import *
from utils.functions import *
from utils.train import train_model
from utils.predict import future


def main(data):
    original_columns = data.columns.tolist()

    # Data Cleaning and Formatting
    data = data_preprocessing(data)

    SET_MONTHLY_DATA = st.sidebar.checkbox('**Convert to Monthly Data?**')

    if SET_MONTHLY_DATA:
        data = to_monthly_data(data)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Analysis & Preprocessing",
         "Train Model",
         "Model Evaluation",
         "Model Components",
         "Forecast",
         "Forecast Evaluation"]
    )

    with tab1:
        st.markdown("""### Your Data Looks like this""")

        plot_data(data, original_columns)
        st.info(
            "**From** " +
            str(data['ds'].to_list()[0].day) + " " +
            str(data['ds'].to_list()[0].month_name()) + " " +
            str(data['ds'].to_list()[0].year) + " " +
            "**To** " + " " +
            str(data['ds'].to_list()[-1].day) + " " +
            str(data['ds'].to_list()[-1].month_name()) + " " +
            str(data['ds'].to_list()[-1].year) + " "
        )

        if SET_MONTHLY_DATA:
            st.download_button(
                "Download Monthly Data", data.to_csv(index=False),
                file_name='monthly_data.csv',
                mime='text/csv'
            )

        st.markdown("### Components")

        comp_data = data.copy()
        plot_components(comp_data)

        SET_NONE_DATA = st.checkbox("**Remove Unusual Event from Data?**")

        backup_data = data.copy()

        if SET_NONE_DATA:
            no_of_event = st.number_input("**How many Events to remove?**")

            if no_of_event != None:
                for e in range(int(no_of_event)):
                    start_d, end_d = st.columns(2)

                    with start_d:
                        start_date = st.date_input(f"**Enter Start Date for event {e + 1}**")
                    with end_d:
                        end_date = st.date_input(f"**Enter End Date for event {e + 1}**")

                    data.loc[(data['ds'] > start_date.strftime("%Y-%m-%d")) & (data['ds'] < end_date.strftime("%Y-%m-%d")), 'y'] = None

                    plot_data(data, original_columns)

        REMOVE_OUTLIERS = st.checkbox("**Remove Outliers?**")

        if REMOVE_OUTLIERS:
            # Outlier Imputation with rolling median
            window_size = st.slider("**Window Side**", 1, 100, 5)
            threshold = st.slider("**Threshold**", 1, 100, 3)

            data['y'] = hampel(data['y'], window_size=window_size, n=threshold, imputation=True)
            st.markdown("""
                #### Your Data After Removing Outliers
            """)

            plot_data(data, original_columns)

            st.download_button(
                "Download Preprocessed Data",
                data.to_csv(index=False),
                file_name='preprocessed_data.csv',
                mime='text/csv'
            )

    with tab2:
        st.markdown("""### Choose Model Your Configuration""")
        model, forecast, step1 = train_model(data)

    with tab3:
        if step1:
            st.markdown("### Model Plot")

            fig = model.plot(forecast)
            a = add_changepoints_to_plot(fig.gca(), model, forecast)
            st.pyplot(fig)
            st.markdown("### Scores")
            st.dataframe(evaluation(backup_data[['y']], forecast[['yhat']]).T)

        else:
            st.warning("Please Train the Model.")

    with tab4:
        if step1:
            st.markdown("### Model Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
        else:
            st.warning("Please Train the Model.")

    with tab5:
        future(data)

    with tab6:
        month_rows = 30
        quarter_rows = 90
        half_rows = 180

        st.markdown("### Forecast Evaluation")

        col1, col2 = st.columns(2)

        with col1:
            actual_file = st.file_uploader('**Upload Actual Data**')

        with col2:
            forecast_file = st.file_uploader('**Upload Forecast Data***')

        MONTHLY_EVAL = st.checkbox("**Is this Monthly Evaluation**")

        if forecast_file is not None and actual_file is not None:
            actual_file = open_file(actual_file)
            actual_file = data_preprocessing(actual_file)

            forecast_file = open_file(forecast_file)
            forecast_file = data_preprocessing(forecast_file)
            forecast_file.columns = ['ds', 'yhat']

            if MONTHLY_EVAL:
                actual_file = to_monthly_data(actual_file)
                month_rows = 1
                quarter_rows = 3
                half_rows = 6

            eval_data = pd.merge(actual_file, forecast_file, how='inner', on='ds')

            st.markdown("#### Evaluation Data")

            plt.figure(figsize=(15, 3))
            plt.title("Actual Vs Predicted", fontsize=20, fontweight="bold")
            plt.plot(eval_data['ds'], eval_data['y'], label="Actual")
            plt.plot(eval_data['ds'], eval_data['yhat'], label="Predicted")
            plt.legend()
            plt.xticks(rotation=30, ha='right')
            plt.xlabel(original_columns[0])
            plt.ylabel(original_columns[1])
            st.pyplot(plt)

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("#### All Data")
                st.dataframe(forecast_evaluation(eval_data).T)

            with col4:
                st.markdown("#### One Month")
                st.dataframe(forecast_evaluation(eval_data.iloc[:month_rows, :]).T)

            with col3:
                st.markdown("#### One Quarter")
                st.dataframe(forecast_evaluation(eval_data.iloc[:quarter_rows, :]).T)

            with col4:
                st.markdown("#### Six Months")
                st.dataframe(forecast_evaluation(eval_data.iloc[:half_rows, :]).T)

            st.markdown("#### Evaluation")

            eval_data['ds'] = eval_data['ds'].dt.date
            eval_data['y'] = eval_data['y'].round()
            eval_data['yhat'] = eval_data['yhat'].round()

            eval_data.columns = ['Date', 'Actual', 'Predicted', 'Absolute Error']
            st.dataframe(eval_data)

            file_name = st.text_input("**To Save, Enter Evaluation File Name**")

            if file_name:
                st.download_button(
                    "Download CSV",
                    eval_data.to_csv(index=False),
                    file_name=file_name + '.csv', mime='text/csv'
                )


if __name__ == '__main__':
    st.title("Forecasting with Facebook's Prophet")
    st.sidebar.markdown("""## Upload Data""")
    PATH = st.sidebar.file_uploader('')

    if PATH is None:
        st.warning("Please Upload Data from Sidebar.")
    else:
        data = open_file(PATH)

        if data is None:
            st.sidebar.error("Cannot open this file, Try Different File.")
        else:
            main(data)

    st.sidebar.info("""Made by @utkarshbelkhede, Intern""")
