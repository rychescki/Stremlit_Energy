import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


@st.cache_data
def load_data(path, sep):
    return next(pd.read_csv(path, sep = str(sep),chunksize=100))

energy_file_path = 'eco2mix-regional-cons-def.csv'
temperature_file_path = 'DST_Energy/temperature_france.csv'
final_data = 'final_df.csv'

def clean_data(df, columns_to_drop=None, filtered_region=None):
    if columns_to_drop:
        df.drop(columns=[col.strip() for col in columns_to_drop if col], inplace=True)

    if filtered_region:
        df = df[df['region (code)'].isin([region.strip() for region in filtered_region if region])]

    return df



page = st.sidebar.radio('Select a page:',
                        ['Project', 'Data Cleaning', 'Data Visualization', 'Our Model', 'Conclusion'])

# Project page
if page == 'Project':
    st.title('France Energy Analysis Project')
    st.write('Welcome to the Stremlit version of our project!')
    st.write('Welcome to our initiative, where we meticulously analyze Frances energy scenario from 2013 to 2022 through advanced analytics and machine learning, particularly highlighting the Gradient Boosting model for its exceptional predictive capabilities. This comprehensive approach not only sheds light on consumption and production patterns but also emphasizes the critical importance of strategic planning and sophisticated modeling in navigating the complexities of energy management. Our project aims to inform future policy-making and contribute to the sustainable and efficient management of energy resources, ensuring a balanced approach to economic growth and environmental stewardship.')


    st.title("Authors:")
    st.write("Matheus Jacobs")
    st.write("Moyeme Abibatou Rosine Sidiguitiebe")
    st.write("Samuel Okoroafor")
    st.write(" ")

    st.title("Advisor:")
    st.write("Tarik Anouar")

    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.write("Data Scientest" )
    st.write("Tour Initiale 1. Terr. Bellini 92800 Puteaux")
    st.write("France")
# Data Cleaning page
elif page == 'Data Cleaning':
    st.title('Data Cleaning')
    st.write('Here is the Data Cleaning section.')

    st.subheader('Energy Data')
    df_energy = load_data(energy_file_path,';')
    df_energy.drop(columns=['TCO Thermique (%)', 'TCH Thermique (%)', 'TCO Nucléaire (%)', 'TCH Nucléaire (%)',
       'TCO Eolien (%)', 'TCH Eolien (%)', 'TCO Solaire (%)',
       'TCH Solaire (%)', 'TCO Hydraulique (%)', 'TCH Hydraulique (%)',
       'TCO Bioénergies (%)', 'TCH Bioénergies (%)', 'Stockage batterie', 'Déstockage batterie',
               'Eolien terrestre', 'Eolien offshore'], inplace = True)
    st.write(df_energy.head(5))

    st.text("The Stockage Batterie, Déstockage batterie, Eolien Terrestre, Eolien Offshore, TCO\nand TCH where drop out, due to a percentage of missing values higher than 75%,\nwhere TCO and TCH are related to the coverage rates and charge rates")


    st.subheader('Temperature Data')
    df_temp = load_data(temperature_file_path, ';')
    df_temperature = df_temp[['Date', 'region (name)', 'region (code)','Température']]
    filtered_region = []

    for element in df_temperature['region (code)'].unique():
        if element in df_energy['Code INSEE région'].unique():
            filtered_region.append(element)
        
    df_temperature = df_temperature[df_temperature['region (code)'].isin(filtered_region)]
    st.write(df_temperature.head(5))
    
    st.text("The temperature per region presented a small % of missing values (0.8).\nHowever, since the data included also costal regions and more days the temperature dataframe was filtered based on the regions\npresent on the energy dataframe and the initial and final date of the same.\nAlso, due to different granularity, we expanded each entry to match the granularity of the energy dataframe and used forward fill to fill the created missing values.")


    st.subheader('Final merged Data')
    merged_df = pd.read_csv('final_df.csv', sep = ',',encoding='utf-8')
    
    st.write("Where the final merged data is:")

    for element in merged_df.columns:
        st.write(f"The column {element} has {merged_df[element].isnull().sum()} missing values")


# Visualizations page 
elif page == 'Data Visualization':
    st.title('Data Visualization')
    st.write('Here you can see different data visualizations.')

    image_figure_1 = Image.open("DST_Energy/Figure1.png")
    st.image(image_figure_1, caption='The upper panel illustrates the Normalized Total Consumption (MW) and Normalized Total Average Temperature from 2013 to 2022, presented in a dual y-axis format where the left y-axis corresponds to consumption (blue) and the right to temperature (red). The consumption and temperature data are normalized to fall between 0 and 1 for comparison on the same scale. The lower panel depicts the Total Consumption of each Energy Type (MW) within the same timeframe, showcasing the varying contributions of different energy sources to the total energy consumption. Each energy type is represented by a distinct color for clear differentiation')


    image_figure_2 = Image.open('DST_Energy/Figure3.png')
    st.image(image_figure_2, caption = "Pie chart representing the distribution of energy consumption in France by type. The largest share is taken by nuclear energy at 72.1$\%$, followed by hydroelectric power at 11.9$\%$, and renewable energy sources including wind (7.5$\%$) solar (1.8$\%$), and bioenergies (1.7$\%$). Thermal energy contributes 5.1$\%$ to the total consumption, while pumped storage energy accounts for the smallest share at 0.9$\%$")


    image_figure_3 = Image.open('DST_Energy/Figure6.png')
    st.image(image_figure_3, caption = 'Map of France color-coded to represent the primary energy source by region. Each color corresponds to the dominant type of energy production or consumption within that region, with nuclear (green), thermal (blue), hydraulic (yellow), and wind (purple).')


    image_figure_4 = Image.open('DST_Energy/Figure5.png')
    st.image(image_figure_4, caption = "Correlation matrix heatmap representing the interrelationships among energy consumption components and generation")



# Modeling page 
elif page == 'Our Model':
    st.title('Gradient Boosting Regressor')
    st.write('This Describes how the GB Regression works.')
    
    st.write('The Gradient Boosting Regressor is a powerful predictive modeling technique that builds an ensemble of decision trees in a sequential way. Each tree is trained to correct the errors of its predecessors, effectively strengthening the models predictive capacity with each iteration.')

    # model parameters input
    st.subheader('Load the full model')

#    n_estimators = st.number_input('Number of estimators', min_value=10, max_value=100, value=50)
#    st.write(f'The model will use {n_estimators} estimators.')

    with open('full_GB_model.pkl', 'rb') as file:
        GB_model = pickle.load(file)
    
    imputed_df = pd.read_csv('final_df.csv', sep = ',')
    st.write(imputed_df.shape)

    X = imputed_df.drop('Consommation (MW)', axis = 1)
    y = imputed_df['Consommation (MW)']

    test_size_value = st.number_input('Percentage of test size:', min_value=0.001, max_value = 0.999, value = 0.2)
    st.write(f"The Model will use {test_size_value*100}% to test the model")


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, random_state=42) 

    GB_model_pred = GB_model.predict(X_test)
    GB_model_pred_train = GB_model.predict(X_train)

    mae = mean_absolute_error(y_test, GB_model_pred)
    mse = mean_squared_error(y_test, GB_model_pred)
    r2 = r2_score(y_test, GB_model_pred)
    r2_train = r2_score(y_train, GB_model_pred_train)

    st.subheader('Model Metrics')
 #   st.write('Model output will be displayed here once the model is ready.')

    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Coefficient of Determination (R² score) on Test set: {r2}")
    st.write(f"Coefficient of Determination (R² score) on Training set: {r2_train}")

    st.subheader('Model Predicitons')

    fig, ax =plt.subplots()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, GB_model_pred, alpha=0.5, c='navy')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted Values')
    
    st.write("Scatter plot between actual and predicted values with our model")
    st.pyplot(plt)


    st.write('Feature Importance of our model')
    plt.clf()
    feature_importances = GB_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center', color = 'navy')
    plt.yticks(range(len(sorted_idx)), X_train.columns[sorted_idx])
    plt.xlabel("Feature importance")
    plt.title("Feature importance for our model")
    st.pyplot(plt)

    plt.clf()

    residuals = y_test - GB_model_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(GB_model_pred, residuals, c='navy')
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals')
    st.pyplot(plt)

# Conclusion page content
elif page == 'Conclusion':
    st.title('Conclusion')
    st.write('Our analysis from 2013 to 2022 offers an in-depth look at Frances energy consumption and production, utilizing advanced data analysis and machine learning, including the Gradient Boosting (GB) model for its predictive accuracy. This model, chosen for its low error rates and ability to handle complex patterns, highlights key factors driving energy use and supports strategic planning. Our findings underline the importance of sophisticated modeling in energy management, aiming for sustainable use and informed policy-making for balancing economic and environmental goals.')

