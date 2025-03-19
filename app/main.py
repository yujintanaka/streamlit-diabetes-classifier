import streamlit as st
import pandas as pd
import pickle as pickle
import numpy as np
from sklearn.manifold import TSNE


def add_prediction(input_dict, data):
    input_data = {}
    for key, value in input_dict.items():
        input_data[key] = [value]
    input_data = pd.DataFrame(input_data)
    column_order = data.columns
    input_data = input_data.reindex(columns=column_order)
    input_data = input_data.drop('Diabetes_binary', axis=1)
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    # adding prediction text
    col1, col2 = st.columns(2)
    with col1:

        if prediction[0] ==  0:
            st.write("<span class='diagnosis normal'>Nondiabetic</span>", unsafe_allow_html=True)
        else:
            st.write("<span class='diagnosis diabetes'>Diabetic or Prediabetic</span>", unsafe_allow_html=True)
    with col2:
        st.page_link("https://github.com/yujintanaka/streamlit-diabetes-classifier", label="About the model", icon="🔗")
    progress_df = pd.DataFrame({"proba":[model.predict_proba(scaled_input)[0][1]]})
    st.data_editor(
        progress_df,
        column_config={
            "proba": st.column_config.ProgressColumn(
                "Probability of being prediabetic or diabetic",
                help="Probability of being prediabetic or diabetic",
                #format="$%f",
                min_value=0,
                max_value=1,
            ),
        },
        hide_index=True,
    )
    return input_data

def add_survey():
    slider_labels = [
        ('BMI',[10,50],'What is your BMI?'),
        ('MentHlth',[0,30],'Thinking about your mental health, which includes stress, depression, and problems with emotions, for how many '
         'days during the past 30 days was your mental health not good?'),
        ('PhysHlth',[0,30],'Thinking about your physical health, which includes physical illness and injury, for how many days during the past '
         '30 days was your physical health not good?')
    ]
    genhlth_map = {
        1:'Excellent',
        2:'Very Good',
        3: 'Good',
        4:'Fair',
        5:'Poor'
    }
    age_map = {
        1: 'Age 18 to 24',
        2: 'Age 25 to 29',
        3: 'Age 30 to 34',
        4: 'Age 35 to 39',
        5: 'Age 40 to 44',
        6: 'Age 45 to 49',
        7: 'Age 50 to 54',
        8: 'Age 55 to 59',
        9: 'Age 60 to 64',
        10: 'Age 65 to 69',
        11: 'Age 70 to 74',
        12: 'Age 75 to 79',
        13: 'Age 80 or older',
    }
    education_map = {
        1:'Never attended school or only kindergarten',
        2:'Grades 1 through 8 (Elementary)',
        3:'Grades 9 through 11 (Some high school)',
        4: 'Grade 12 or GED (High school graduate) ',
        5:'College 1 year to 3 years (Some college or technical school) ',
        6:'College 4 years or more (College graduate)'
    }
    income_map = {
        1:'Less than $10,000',
        2:'Less than $15,000 ($10,000 to less than $15,000)',
        3: 'Less than $20,000 ($15,000 to less than $20,000)',
        4:'Less than $25,000 ($20,000 to less than $25,000)',
        5: 'Less than $35,000 ($25,000 to less than $35,000)',
        6:'Less than $50,000 ($35,000 to less than $50,000)',
        7:'Less than $75,000 ($50,000 to less than $75,000)',
        8:'$75,000 or more'
    }
    dropdown_labels = [
        ('Age', age_map, 'Age:'),
        ('Education', education_map, 'What is the highest grade or year of school you completed?'),
        ('Income', income_map, 'What is your annual household income from all sources?'),
        ('GenHlth',genhlth_map,'Would you say that in general your health is:')
    ]
    binary_labels = {
        'HighBP': 'Have you been told you have high blood pressure by a doctor, nurse, or other health professional?',
        'HighChol': 'Have you ever been told by a doctor, nurse or other health professional that your blood cholesterol is high?', 
        'CholCheck': 'Have you checked your cholesterol in the past 5 years?',
        'Smoker': 'Have you smoked more than 100 cigarettes? (5 Packs)',
        'Stroke': 'Have you ever suffered from stroke?',
        'HeartDiseaseorAttack': 'Have you ever had coronary heart disease (CHD) or myocardial infarction', 
        'PhysActivity':'Have you had physical activity in the past 30 days?', 
        'Fruits': 'Do you consume fruit 1 or more times a day?', 
        'Veggies': 'Do you consume vegetables 1 or more times a day?',
        'HvyAlcoholConsump': 'Are you a heavy drinker?(adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)', 
        'AnyHealthcare': 'Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMO', 
        'NoDocbcCost': 'Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?', 
        'DiffWalk': 'Do you have serious difficulty walking or climbing stairs?',
    }
    
    input_dict = {}

    binary_map = {1:"yes",0: "no"}
    for feature, question in binary_labels.items():
        st.write(question)
        input_dict[feature] = st.selectbox(
            label= feature,
            key = feature,
            label_visibility='hidden',
            options= binary_map.keys(),
            format_func= lambda option: binary_map[option],

        )
        st.write('')
    sex_map = {1:'Male',0:'Female'}
    st.write('Sex: ')
    input_dict['Sex'] = st.selectbox(
        label= 'Sex',
        key = 'Sex',
        label_visibility='hidden',
        options= sex_map.keys(),
        format_func= lambda option: sex_map[option],
    )
    st.write('')

    for feature, mapping, question in dropdown_labels:
        st.write(question)
        input_dict[feature] = st.selectbox(
            label= feature,
            key = feature,
            label_visibility='hidden',
            options= mapping.keys(),
            format_func= lambda option: mapping[option],
        )
        st.write('')
    for feature, range, question in slider_labels:
        st.write(question)
        input_dict[feature] = st.slider(
            label=feature,
            key=feature,
            label_visibility='hidden',
            min_value=range[0],
            max_value=range[1],
            value=20
        )
        st.write('')
    return input_dict

def get_tsne(data_point):

    # prepare data point
    data_point['Diabetes_binary'] = 2

    #prepare Data and add datapoint

    tsne_df = pd.read_csv('data/small_df.csv')
    tsne_df = pd.concat([tsne_df, data_point], ignore_index=True) 
    features = tsne_df.drop(columns=['Diabetes_binary'])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Add the t-SNE results to the DataFrame
    tsne_df['tsne_x'] = tsne_results[:, 0]
    tsne_df['tsne_y'] = tsne_results[:, 1]

    # Convert Diabetes_binary to string to make categorical
    tsne_df['Diabetes_binary'] = tsne_df['Diabetes_binary'].map({0: 'NonDiabetic', 1:'Prediabetic or Diabetic', 2: 'Your Survey Results' })
    # saves the output so the plot will be rendered without compute
    st.session_state.tsne_df = tsne_df


def add_sidebar(input_data):
    data = pd.read_csv('data/small_df.csv')
    data_point = add_prediction(input_data,data)
    st.title('TSNE Visualization')
    st.write('Visualize how similar your survey results are to other people')
    if st.button('Run TSNE'):
        get_tsne(data_point)
    if 'tsne_df' in st.session_state:
        if isinstance(st.session_state.tsne_df, pd.DataFrame):
            st.scatter_chart(data=st.session_state.tsne_df, x='tsne_x', y='tsne_y', x_label='', y_label='', color='Diabetes_binary')



def main():
    st.set_page_config(
        page_title='Diabetes Risk Factor Classification',
        layout='wide',
        page_icon=':female-doctor:',
        initial_sidebar_state='expanded'
    )
    st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 700px !important; 
        }
        .st-emotion-cache-a6qe2i  {
            margin-top: -25px;
        }
        .diagnosis.normal {
            background-color: blue;
        }
        .diagnosis.diabetes {
            background-color: red;
        }
        .diagnosis {
            color: #fff;
            padding: 0.2em 0.5rem;
            border-radius: 0.5em;
            font-size: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    # init session state
    if 'tsne_df' not in st.session_state:
        st.session_state.tsne_df = None
    st.title('Diabetes Risk Factors Classification Model')
    st.write('Fill out the survey to see how similar you are to a diabetic person')
    input_data = add_survey()
    with st.sidebar:
        add_sidebar(input_data)


if __name__ == '__main__':
    main()
