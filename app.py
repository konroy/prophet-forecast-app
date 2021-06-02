import streamlit as st
from streamlit import caching
import pandas as pd
import numpy as np

import pystan
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import json
from fbprophet.serialize import model_to_json, model_from_json

import altair as alt
import plotly as plt
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.figure_factory as ff
import base64
import itertools
from datetime import datetime

st.set_page_config(page_title ="Forecast App",
                    initial_sidebar_state="collapsed",
                    page_icon="🔮")


tabs = ["Application","About"]
page = st.sidebar.radio("Tabs",tabs)


@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner= True)

def load_csv():
    
    df_input = pd.DataFrame()  
    df_input=pd.read_csv(input,sep=None ,engine='python', encoding='utf-8',
                            parse_dates=True,
                            infer_datetime_format=True)
    return df_input

def prep_data(df):

    df_input = df.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    st.markdown("The selected date column is now labeled as **ds** and the values columns as **y**")
    df_input = df_input[['ds','y']]
    df_input =  df_input.sort_values(by='ds',ascending=True)
    return df_input

if page == "Application":

	with st.sidebar:
		if st.button(label='Clear cache'):
			caching.clear_cache()

	st.title('📈🧙🏻 Automated Time Series Forecasting')
	st.write('This app enables you to generate time series forecast without any dependencies.')
	st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")
	caching.clear_cache()
	df =  pd.DataFrame()   

	st.subheader('Data loading 🏋️')
	st.write("Import a time series csv file.")
	with st.beta_expander("Data format"): 
		st.write("The dataset can contain multiple columns but you will need to select a column to be used as dates and a second column containing the metric you wish to forecast. The columns will be renamed as **ds** and **y** to be compliant with Prophet. Even though we are using the default Pandas date parser, the ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric.")

	input = st.file_uploader('')

	if input is not None:
		with st.spinner('Loading data..'):
			df = load_csv()

			st.write('Columns:')
			st.write(list(df.columns))
			columns = list(df.columns)

			col1,col2 = st.beta_columns(2)
			with col1:
				date_col = st.selectbox("Select date column:",index=0,options=columns,key='date')
			with col2:
				metric_col = st.selectbox("Select metric column:",index=1,options=columns,key='values')
			
			df = prep_data(df)
			output = 0

		if st.checkbox('Chart data', key='show'):
			with st.spinner('Plotting data...'):
				col1,col2 = st.beta_columns(2)
				with col1:
					st.dataframe(df)

				with col2:
					st.write("Description of dataframe: ")
					st.write(df.describe())
			try:
				line_chart = alt.Chart(df).mark_line().encode(
					x = 'ds:T',
					y = "y:Q", tooltip = ['ds:T','y']
					).properties(title="Time series preview").interactive()
				st.altair_chart(line_chart,use_container_width=True)
			except:
				st.line_chart(df['y'],use_container_width =True,height = 300)

		with st.beta_container():
			st.subheader("Parameters configuration 🛠️")
			st.write('In this section you can modify the algorithm settings.')

			with st.beta_expander("Horizon"):
				periods = st.number_input('Select the horizon (days to predict) to forecast', min_value = 1, max_value = 730, value = 90)

			with st.beta_expander("Seasonality"):
				st.markdown("The default seasonality is additive, need domain knowledge to know which is best choice. For more informations visit the [documentation](https://facebook.github.io/prophet/docs/multiplicative_seasonality.html)")
				seasonality = st.radio(label='Seasonality',options=['additive','multiplicative'])
			
			with st.beta_expander("Trend components"):
				st.write("Add or remove components: ")
				daily = st.checkbox('Daily')
				weekly = st.checkbox('Weekly')
				monthly = st.checkbox('Monthly')
				yearly = st.checkbox('Yearly')
			
			with st.beta_expander("Growth Model"):
				st.write("Prophet uses a linear growth model by default")
				st.markdown("For more infomation check the [documentation](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth)")
				
				growth = st.radio(label='Growth model',options=['linear','logistic'])

				if growth == 'linear':
					growth_settings = {
						'cap':1,
						'floor':0
					}
					cap=1
					floor=1
					df['cap']=1
					df['floor']=0

				if growth == 'logistic':
					st.info('Configure saturation')

					cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
					floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
					if floor > cap:
						st.error('Cap must be larger than floor!')
						growth_settings={}
					if floor == cap:
						st.error('Cap must be larger than floor!')
					else:
						growth_settings = {
							'cap':cap,
							'floor':floor
						}
						df['cap']=cap
						df['floor']=floor
			
			with st.beta_expander('Holidays'):
				countries = ['Country name', 'Malaysia']

				with st.beta_container():
					years = [2021]
					selected_country = st.selectbox(label="Select country",options=countries)

					if selected_country == "Country name":
						holidays = False

					holidays = st.checkbox('Add country holidays to the model')

			with st.beta_expander("Hyperparameter"):
				st.write('This section it is for tuning the scaling coefficients.')

				seasonality_scale_values = [0.1,1.0,5.0,10.0]
				changepoint_scale_values = [0.01,0.1,0.5,1.0]

				st.write('The changepoint prior scale determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints')
				changepoint_scale = st.select_slider(label = 'Changepoint prior scale', options = changepoint_scale_values)

				st.write('The seasonality change point controls the flexibility of the seasonality.')
				seasonality_scale = st.select_slider(label = 'Seasonality prior scale', options = seasonality_scale_values)
		
				st.markdown("""For more information read the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)""")

		with st.beta_container():
			st.subheader("Forecast 🔮")
			st.write("Fit the model on the data and generate future prediction.")
			st.write("Load a time series to activate.")
			if input is not None:
				if st.checkbox("Initialize model (Fit)",key="fit"):
					if len(growth_settings)==2:
						
						m = Prophet(
								seasonality_mode = seasonality,
								daily_seasonality=daily,
								weekly_seasonality=weekly,
								yearly_seasonality=yearly,
								growth=growth,
								changepoint_prior_scale=changepoint_scale,
								seasonality_prior_scale= seasonality_scale
							)

						if holidays:
							m.add_country_holidays(country_name=selected_country)

						if monthly:
							m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)

						with st.spinner('Fitting the model...'):
							m = m.fit(df)
							future = m.make_future_dataframe(periods=periods,freq='D')
							future['cap'] = cap
							future['floor'] = floor
							future = future.dropna()
							st.write("The model will produce forecast up to ", future['ds'].max())
							st.success('Model fitted successfully')
					else:
						st.warning('Invalid configuration')
				
				if st.checkbox("Generate Forecast/Prediction",key="predict"):
					try:
						with st.spinner("Forecasting/Predicting..."):
							forecast = m.predict(future)
							st.success('Prediction generated successfully')
							st.dataframe(forecast)
							fig1 = m.plot(forecast)
							st.write(fig1)
							output = 1

							if growth == 'linear':
								fig2 = m.plot(forecast)
								a = add_changepoints_to_plot(fig2.gca(), m, forecast)
								st.write(fig2)
								output = 1
					except:
						st.warning("You need to train the model!")

				if st.checkbox('Show components'):
					try:
						with st.spinner("Loading components..."):
							fig3 = m.plot_components(forecast)
							st.write(fig3)
					except:
						st.warning("You need to do forecast generation first!")


		with st.beta_container():
			st.subheader('Model validation 🧪')
			st.write("In this section it is possible to do cross-validation of the model.")

			with st.beta_expander("Explanation"):
				st.markdown("""The Prophet library enables us to divide our historical data into training data and testing data for cross validation. The main concepts for cross validation with Prophet are:""")
				st.write("Training data (initial): The amount of data set aside for training. The parameter in the API is called initial.")
				st.write("Horizon: The data set aside for validation.")
				st.write("Cutoff (period): a forecast is made for every observed point between cutoff and cutoff + horizon.""")

			with st.beta_expander("Cross Validation "):
				initial = st.number_input(value=365, label="initial", min_value=30, max_value=1096)
				initial = str(initial) + 'days'

				period = st.number_input(value=90, label="period", min_value=1, max_value=365)
				period = str(period) + 'days'

				horizon = st.number_input(value=90, label="horizon", min_value=30, max_value=366)
				horizon = str(horizon) + 'days'

				st.write(f"Here we do cross-validation to assess prediction performance on a horizon of **{horizon}** days, starting with **{initial}** days of training data in the first cutoff and then making predictions every **{period}**.")
				st.markdown("For more information read the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)")

			with st.beta_expander("Metrics"):
				if input is not None:
					if output == 1:
						metrics = 0
						if st.checkbox("Calculate Metrics"):
							with st.spinner("Cross Validating..."):
								try:
									df_cv = cross_validation(
											m,
											initial = initial,
											period = period,
											horizon = horizon,
											parallel = 'processes'
										)
									df_p = performance_metrics(df_cv)
									st.write(df_p)
									metrics = 1
								except:
									st.write("Invalid configurations!")
									metrics = 0

								st.markdown("Metrics Definition")
								st.write("MSE: Mean Absolute Error")
								st.write("MAE: Mean Average Error")
								st.write("MAPE: Mean Average Percentage Error")
								st.write("MSE: Mean Absolute Error")
								st.write("MDAPE: Median Average Percentage Error")

								if metrics == 1:
									metrics = ['Choose a metric','mse','rmse','mae','mape','mdape','coverage']
									selected_metric = st.selectbox('Select metric to plot', options=metrics)
									if selected_metric != metrics[0]:
										fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
										st.write(fig4)
						else:
							st.write("Create a forecast to see metrics.")

		with st.beta_container():
			st.subheader('Hyperparameter Tuning 🧲')
			st.write("In this section it is possible to find the best combination of hyperparamenters.")
			st.markdown("""For more informations visit the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)""")
			
			param_grid = {
				'changepoint_prior_scale' : [0.01, 0.1, 0.5, 1.0],
				'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0],
			}

			all_params = [dict(zip(param_grid.keys(),v)) for v in itertools.product(*param_grid.values())]
			rmses = []

			if input is not None:
				if output == 1:
					if st.button("Optimize hyperparamenters"):
						with st.spinner("Finding best combination. Please wait..."):
							try:
								for params in all_params:
									m = Prophet(**params).fit(df)
									df_cv = cross_validation(
										m, 
										initial=initial,
										period=period,
										horizon=horizon,
										parallel="processes"
										)
									df_p = performance_metrics(df_cv,rolling_window=1)
									rmses.append(df_p['rmse'].values[0])
							except:
								for params in all_params:
									m = Prophet(**params).fit(df)
									df_cv = cross_validation(
										m,
										initial=initial,
										period=period,
										horizon=horizon,
										parallel="threads"
										)
									df_p = performance_metrics(df_cv, rolling_window=1)
									rmses.append(df_p['rmse'].values[0])

						#find best params
						tuning_results = pd.DataFrame(all_params)
						tuning_results['rmse'] = rmses
						st.write(tuning_results)

						best_params = all_params[np.argmin(rmses)]

						st.write('The best parameter combination is: ')
						st.write(best_params)
						st.write(" You may repeat the process using these parameters in the configuration section 2")
				else:
					st.write("Create a model to optimize")    

		with st.beta_container():
			st.subheader('Export results ✨')
			st.write("Finally you can export your result forecast, model configuration and evaluation metrics.")

			if input is not None:
				if output == 1:
					col1, col2, col3, col4 = st.beta_columns(4)

					with col1:
						if st.button('Export forecast (csv.)'):
							with st.spinner('Exporting...'):
								export_forecast = pd.DataFrame(forecast[['ds','yhat_lower','yhat','yhat_upper']])
								st.write(export_forecast.head())
								export_forecast = export_forecast.to_csv(decimal=',')
								b64 = base64.b64encode(export_forecast.encode()).decode()
								href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv" >Download CSV File</a> '
								st.markdown(href, unsafe_allow_html = True)

					with col2:
						if st.button("Export model metrics (csv.)"):
							try:
								df_p = df_p.to_csv(decimal=',')
								b64 = base64.b64encode(df_p.encode()).decode()
								href = f'<a href="data:file/csv;base64,{b64}" download="metrics.csv" >Download CSV File</a> '
								st.markdown(href, unsafe_allow_html = True)
							except:
								st.write("No metric to export")

					with col3:
						if st.button("Save model config (.json) in memory"):
							with open('serialized_model.json', 'w') as fout:
								json.dump(model_to_json(m),fout)

					with col4:
						if st.button('Clear cache memory'):
							caching.clear_cache()

				else:
					st.write('Generate a forecast to download.')	

			# data = pd.read_csv(df)
			# st.write(data)
			# ds = st.selectbox("Select date column: ", ('test','test2'))
			# y = st.selectbox("Select value column: ", ('test','test2'))

