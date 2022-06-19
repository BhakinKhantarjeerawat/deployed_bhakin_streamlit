import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib, os

#########################
##### get value #########
#########################
def get_value(val, my_dicts):
	for key, value in my_dicts.items():
		if val == key:
			return value


#########################
##### get key ###########
#########################
def get_key(val, my_dicts):
	for key, value in my_dicts.items():
		if val == value:
			return key


#########################
##### Load Model ########
#########################
def load_prediction_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
	return loaded_model



#########################
##### main function #####
#########################
def main():
	# """Deploying Streamlit App with Docker"""
	# st.title("Streamlit App")

	html_templ = '''
	<div style='background-color:magenta; padding:10px;'>
	<h2 style='color:white'>Contraceptive Prediction</h2>
	</div>
	'''
	st.markdown(html_templ, unsafe_allow_html=True)
	st.markdown("Beta Version By Bhakin Khantarjeerawat, created by using streamlit on Sep 2020 and adapted from one of Jesse Abbe'Projects")

	activity = ["Descriptive","Predictive"]
	choice = st.sidebar.selectbox('Choose Analytics Type', activity)

	########################
	##### Descriptinve #####
	########################
	if choice == 'Descriptive':
		st.subheader("Exploratory Data Analysis")

		df = pd.read_csv('data_cmc/cmc_dataset.csv')
		if st.checkbox('Preview dataset'):
			number = st.number_input('Select Number of Rows to View', min_value=1)
			st.dataframe(df.head(number))
		if st.checkbox('Shape of Dataset'):
			st.write(df.shape)
			data_dim = st.radio('Show Dimensions By', ('Rows', 'Columns'))
			if data_dim == 'Rows':
				st.text('Numbers of Rows:')
				st.write(df.shape[0])
			if data_dim == 'Columns':
				st.text('Numbers of Columns:')
				st.write(df.shape[1])

		if st.checkbox('Select Columns'):
			all_columns = df.columns.tolist()
			selected_columns = st.multiselect('Select Columns', all_columns)
			new_df = df[selected_columns]
			st.dataframe(new_df)

		if st.checkbox('Summary of Dataset'):
			st.write(df.describe())

		if st.checkbox('Value Counts'):
			st.text('Value Counts by Target')
			st.write(df.iloc[:,-1].value_counts())

		st.subheader('Data Visualization')
		if st.checkbox('Correlatioin Plot (by Seaborn)'):
			st.write(sns.heatmap(df.corr()))
			st.pyplot()
		if st.checkbox('Pie Chart'):
			# if st.checkbox('Generate Pie Chart'):
				st.write(df.iloc[:, -1].value_counts().plot.pie(autopct='%1.1f%%'))
				st.pyplot()
		# if st.checkbox('Plot Value Counts'):
		# 	vlc_df = df.iloc[:-1]
		# 	vlc_sr = pd.Series(vlc_df).value_counts()
		# 	st.write(vlc_sr.plot(kind='bar'))
		# 	st.pyplot()
		if st.checkbox('Plot Value Counts by Columns'):
			st.text('Value Counts by Target/Class')

			all_columns_names = df.columns.tolist()
			primary_col = st.selectbox('Select Primary Columns', all_columns_names)
			selected_column_names = st.multiselect('Select Multiple Columns', all_columns_names)
			if st.checkbox('Plot (Note: Primary Columns AND Multiple Columns must be selected First)'):
				st.text('Generating Plot for: {} and {}'.format(primary_col, selected_column_names))
				if selected_column_names:
					vc_plot = df.groupby(primary_col)[selected_column_names].count()
				# else:
				# 	vc_plot = df.iloc[:,-1]
				st.write(vc_plot.plot(kind='bar'))
				st.pyplot()

	######################
	##### Predictive #####
	######################
	if choice == 'Predictive':
		st.subheader("Predictive Aspect")

		age = st.slider('Select Age', 16, 60)
		wife_education = st.number_input('Select Wife Education Level (Lowest(1) to Highest(4))', 1,4)
		husband_education = st.number_input('Select Husband Education Level (Lowest(1) to Highest(4))', 1,4)
		number_of_children_ever_born = st.number_input('Number of Children Ever Born(Lowest(0)', 1)

		wife_reg = {'Non-Religios':0, 'Religios':1}
		choice_wife_reg = st.radio('Wife Religion', tuple(wife_reg.keys()))
		result_wife_reg = get_value(choice_wife_reg, wife_reg)

		wife_working = {'Yes':0, 'No':1}
		choice_wife_working = st.radio('Wife Working', tuple(wife_working.keys()))
		result_wife_working = get_value(choice_wife_working, wife_working)

		husband_occupation = st.number_input('Husband Occupation Level (Lowest=1, Highest=4)',1,4)
		standard_of_living = st.slider('Standard of Living (Lowest=1, Highest=4)', 1,4)

		media_exposure = {'Good':1, 'Not-Good':0, }
		choice_of_media_exposure = st.radio('Media Exposure', tuple(media_exposure.keys()))
		result_media_exposure = get_value(choice_of_media_exposure, media_exposure)

		#### result ###
		st.subheader('Summary of Input Data:')
		results = [age, wife_education, husband_education, number_of_children_ever_born, result_wife_reg, result_wife_working, husband_occupation, standard_of_living, result_media_exposure]
		prettified_result = {'age':age,
							'wife_education':wife_education,
							'husband_education':husband_education,
							'number_of_children_ever_born': number_of_children_ever_born,
							'result_wife_reg':result_wife_reg,
							'wife_reg':choice_wife_reg,
							'result_wife_working':result_wife_working,
							'wife_working':choice_wife_working,
							'husband_occupation':husband_occupation,
							'standard_of_living':standard_of_living,
							'result_media_exposure':result_media_exposure,
							'media_exposure':choice_of_media_exposure,}
		sample_data = np.array(results).reshape(1,-1)
		st.write(sample_data)
		st.info(sample_data)
		st.json(prettified_result)

		st.subheader('Predictive Models')
		prediction_label = {'No Use':1, 'Longterm':2, 'Shortterm':3}
		if st.checkbox('Make A Prediction'):
			predictive_model = load_prediction_model('models/cmcModel.pkl')
			predictive_result = predictive_model.predict(sample_data)
			st.write(predictive_result)
			final_result = get_key(predictive_result, prediction_label)
			st.success(final_result)




if __name__ == '__main__':
	main()
