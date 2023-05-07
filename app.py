import pandas_profiling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import streamlit as st
from streamlit_pandas_profiling import st_profile_report


# creating a function for Prediction
# Loading model to compare the results
def chooseModel(modelName):
    loaded_model = pickle.load(open(modelName+'.pkl','rb'))
    return loaded_model

def diabetes_prediction(model_choice,
                Gender,
                age,
                arrival_delay_in_minutes,
                baggage_handling,
                checkin_service,
                cleanliness,
                customer_class,
                customer_type,
                departure_arrival_time_convenient,
                departure_delay_in_minutes,
                ease_of_online_booking,flight_distance,
                food_and_drink,
                gate_location,
                inflight_entertainment,
                inflight_service,
                inflight_wifi_service,
                leg_room_service,
                onboard_service,
                online_boarding,
                seat_comfort,
                type_of_travel):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.array([[Gender,
                age,
                arrival_delay_in_minutes,
                baggage_handling,
                checkin_service,
                cleanliness,
                customer_class,
                customer_type,
                departure_arrival_time_convenient,
                departure_delay_in_minutes,
                ease_of_online_booking,
                flight_distance,
                food_and_drink,
                gate_location,
                inflight_entertainment,
                inflight_service,
                inflight_wifi_service,
                leg_room_service,
                onboard_service,
                online_boarding,
                seat_comfort,
                type_of_travel]]).astype(np.float64)
    print(input_data_as_numpy_array)
    # # reshape the array as we are predicting for one instance
    # input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    # b = np.array(input_data_reshaped, dtype=float)  # convert using numpy
    # c = [float(i) for i in input_data_reshaped]


    loaded_model = chooseModel(model_choice)
    prediction = loaded_model.predict(input_data_as_numpy_array)

    print("pred",prediction)
    return prediction




with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Customer Satisfaction Prediction")
    choice = st.radio("Navigation", ["Profiling","Modelling", "EDA"])
    st.info("This project application helps you build and explore your data.")

def visualize():
    df = pd.read_csv("./airline_passenger_satisfaction.csv")
    df.head()
    # Analysing categorical variables first
    # ['Gender', 'customer_type', 'type_of_travel', 'customer_class', 'satisfaction']

    #  https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8
#line 26
    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='Gender', data=df, palette='autumn');
    ax = sns.countplot(x='Gender', data=df, hue='satisfaction', palette='viridis')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    fig, axes = plt.subplots(figsize=(5, 5))
    ax = sns.countplot(x='customer_type', data=df, palette='autumn');
    ax = sns.countplot(x='customer_type', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    fig, axes = plt.subplots(figsize=(5, 5))
    ax = sns.countplot(x='type_of_travel', data=df, palette='autumn');
    ax = sns.countplot(x='type_of_travel', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='customer_class', data=df, palette='autumn');
    ax = sns.countplot(x='customer_class', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

#line 27
    #  https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8

    #  analyzing numerical variable
    # ['age', 'flight_distance', 'departure_delay_in_minutes',  'arrival_delay_in_minutes']

    # 'inflight_wifi_service', 'departure_arrival_time_convenient', 'ease_of_online_booking','gate_location',

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='inflight_wifi_service', data=df, palette='autumn');
    ax = sns.countplot(x='inflight_wifi_service', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='departure_arrival_time_convenient', data=df, palette='autumn');
    ax = sns.countplot(x='departure_arrival_time_convenient', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='ease_of_online_booking', data=df, palette='autumn');
    ax = sns.countplot(x='ease_of_online_booking', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='gate_location', data=df, palette='autumn');
    ax = sns.countplot(x='gate_location', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    # 'food_and_drink','online_boarding', 'seat_comfort',

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='food_and_drink', data=df, palette='autumn');
    ax = sns.countplot(x='food_and_drink', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='online_boarding', data=df, palette='autumn');
    ax = sns.countplot(x='online_boarding', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='seat_comfort', data=df, palette='autumn');
    ax = sns.countplot(x='seat_comfort', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    #  'inflight_entertainment', 'onboard_service','leg_room_service', 'baggage_handling',  'checkin_service',  'inflight_service',

    #  'cleanliness',

    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='inflight_entertainment', data=df, palette='autumn');
    ax = sns.countplot(x='inflight_entertainment', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='onboard_service', data=df, palette='autumn');
    ax = sns.countplot(x='onboard_service', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='leg_room_service', data=df, palette='autumn');
    ax = sns.countplot(x='leg_room_service', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='baggage_handling', data=df, palette='autumn');
    ax = sns.countplot(x='baggage_handling', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='checkin_service', data=df, palette='autumn');
    ax = sns.countplot(x='checkin_service', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='inflight_service', data=df, palette='autumn');
    ax = sns.countplot(x='inflight_service', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


    fig, axes = plt.subplots(figsize=(5, 5))
    sns.countplot(x='cleanliness', data=df, palette='autumn');
    ax = sns.countplot(x='cleanliness', data=df, hue='satisfaction', palette='viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)


# line 28
    numerical_variables = df.select_dtypes(np.number).columns.to_list()
    for col in numerical_variables:
        fig, ax = plt.subplots(figsize=(20, 5))
        sns.boxplot(x=df[col], palette='autumn')
        st.pyplot(fig)

# line 29
    # create graph
    sns.displot(x = "departure_delay_in_minutes", data = df, hue ="satisfaction",log_scale = False ,height = 10,palette ="autumn"  )
    plt.title("departure_delay_in_minutes results by Flight Distance")
    plt.show()

#line 31
    continuos_variable = ['age', 'flight_distance', 'departure_delay_in_minutes',  'arrival_delay_in_minutes']
    for col in continuos_variable:
        fig, ax = plt.subplots(figsize = (20, 5))
        sns.histplot(df[col],palette= 'autumn', color='orange')
        st.pyplot(fig)

#line 32
    # create graph
    sns.displot(x="age", data=df, hue="satisfaction", height=8, palette="autumn")
    plt.show()
    st.pyplot(fig)

    sns.displot(x="flight_distance", data=df, hue="satisfaction", height=8, palette="autumn")
    plt.show()
    st.pyplot(fig)

#line 33
# create graph
    g = sns.FacetGrid(df, col = "type_of_travel", hue ="satisfaction", height = 8, palette ="autumn" )
    g.map(sns.histplot, "flight_distance",  )
    g.add_legend()
    plt.show()
    st.pyplot(fig)


#line 34
    fig, axes = plt.subplots(figsize = (5, 5))
    ax = sns.countplot(x = 'customer_class', data = df, palette= 'autumn');
    ax =sns.countplot(x = 'customer_class', data = df, hue= 'type_of_travel', palette= 'viridis');
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

#line 35
    # create graph
    sns.displot(x='arrival_delay_in_minutes', data=df, hue="satisfaction", palette="autumn")
    plt.xticks([0, 50, 100, 150, 200, 250, 300])
    plt.yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    plt.xlim(0, 500)
    plt.ylim(0, 1500)
    plt.show()
    st.pyplot(fig)

    sns.displot(x='departure_delay_in_minutes', data=df, hue="satisfaction", palette="autumn")
    plt.xticks([0, 50, 100, 150, 200, 250, 300])
    plt.yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    plt.xlim(0, 500)
    plt.ylim(0, 1500)
    plt.show()
    st.pyplot(fig)

#line 36
    # specify size of heatmap
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=True)
    st.pyplot(fig)

#line 37
    # Departure Delay and Arrival delay are heavily correlated with each other, therefore one of them should be dropped
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x='departure_delay_in_minutes', y='arrival_delay_in_minutes', data=df, palette="autumn")
    plt.show()
    st.pyplot(plt)


if choice == "Profiling":
    df = pd.read_csv("./airline_passenger_satisfaction.csv")
    pr = df.profile_report()
    st_profile_report(pr)

if choice == "EDA":
   visualize()

if choice=="Modelling":
    # giving a title
    rating = [0,1,2,3,4,5]
    st.title('Airline Passenger satisfaction')

    import streamlit as st

    col1, col2,col3 = st.columns(3)

    with col1:
        Gender = st.selectbox('Gender', ("Male", "Female"))
        if Gender == "Male":
            Gender = 1
        else:
            Gender = 0

    with col2:
        age = st.number_input('Age', min_value=18, max_value=110)

    with col3:
        arrival_delay_in_minutes = st.number_input('Arrival Delay in minutes',min_value=0)

    with col1:
        baggage_handling = st.selectbox('Baggage Handling',rating)

    with col2:
        checkin_service = st.selectbox('Checkin Service',rating)

    with col3:
        cleanliness = st.selectbox('Cleanliness',rating)

    with col1:
        customer_class = st.selectbox('Customer Class',("Eco","Eco Plus", "Business"))
        if customer_class == "Eco":
            customer_class = 1
        elif customer_class=="Eco Plus":
            customer_class = 0
        else:
            customer_class=2

    with col2:
        customer_type = st.selectbox('Customer Type',("Loyal","Disloyal"))
        if customer_type == "Loyal":
            customer_type = 1
        else:
            customer_type = 0

    with col3:
        departure_arrival_time_convenient = st.number_input('Departure Arrival Time Convenient',min_value=0)

    with col1:
        departure_delay_in_minutes = st.number_input('Departure Delay in minutes',min_value=0)

    with col2:
        ease_of_online_booking = st.selectbox('Ease of online booking',rating)

    with col3:
        flight_distance = st.number_input('Flight Distance',min_value=200)

    with col1:
        food_and_drink = st.selectbox('Food & Drink',rating)

    with col2:
        gate_location = st.selectbox('Gate Location',rating)

    with col3:
        inflight_entertainment = st.selectbox('Inflight Entertainment',rating)

    with col1:
        inflight_service = st.selectbox('Inflight Service',rating)

    with col2:
        inflight_wifi_service = st.selectbox('Inflight Wifi Service',rating)

    with col3:
        leg_room_service = st.selectbox('Leg Room Service',rating)

    with col1:
        onboard_service = st.selectbox('Onboard Service',rating)

    with col2:
        online_boarding = st.selectbox('Online Boarding',rating)

    with col3:
        seat_comfort = st.selectbox('Seat Comfort',rating)

    with col1:
        type_of_travel = st.selectbox('Type of Travel',("Business","Personal"))
        if type_of_travel == "Business":
            type_of_travel = 1
        else:
            type_of_travel = 0

    # creating a button for Prediction

    all_ml_models = ["Logistic Regression",  "SVM","Naive Bayes","XG Boost","Random Forest"]
    model_choice = st.radio("Select Model", all_ml_models)
    res=''
    if st.button('Customer Satisfaction Result'):


        res = diabetes_prediction(model_choice,Gender,
                age,
                arrival_delay_in_minutes,
                baggage_handling,
                checkin_service,
                cleanliness,
                customer_class,
                customer_type,
                departure_arrival_time_convenient,
                departure_delay_in_minutes,
                ease_of_online_booking,flight_distance,
                food_and_drink,
                gate_location,
                inflight_entertainment,
                inflight_service,
                inflight_wifi_service,
                leg_room_service,
                onboard_service,
                online_boarding,
                seat_comfort,
                type_of_travel)
        # code for Prediction
        print("res",res)
        if res==[1]:
            st.metric(label="Satisfaction Level :blush::blush::blush::blush::blush:", value="Satisifed ", delta="Happy with the service")
        else:
            st.metric(label="Satisfaction Level :rage::rage::rage::rage::rage::rage:", value="Not Satisfied ", delta="Unhappy with the service")


