import streamlit as st
import pandas as pd
import pickle

rfc=pickle.load(open('rfc.pkl','rb'))
std=pickle.load(open('std.pkl','rb'))
option=['Yes','No']

st.title('Travel Insurance Predictor')

col1,col2=st.columns(2)
with col1:
    Age=st.number_input('Age',min_value=16,step=1,value=25)
with col2:
    Employment_Type = st.selectbox('Employment Type',['Government Sector','Private Sector/Self Employed'],index=1)

col3,col4=st.columns(2)
with col3:
    GraduateOrNot=st.selectbox('Graduate or Not',option,index=0)
with col4:
    AnnualIncome=st.number_input('Annual Income',min_value=250000,step=5000,value=1500000)

col5,col6=st.columns(2)
with col5:
    FamilyMembers=st.number_input('Family Members',min_value=0,step=1,value=2)
with col6:
    ChronicDiseases=st.selectbox('Chronic Disease',option,index=0)

col7 ,col8=st.columns(2)
with col7:
    FrequentFlyer=st.selectbox('Frequent Flyer',option,index=0)
with col8:
    EverTravelledAbroad = st.selectbox('Ever Travelled Abroad', option,index=0)

def pre_processing(pred):
    pred[['Age', 'AnnualIncome', 'FamilyMembers']] = std.transform(pred[['Age', 'AnnualIncome', 'FamilyMembers']])
    pred['GraduateOrNot'].replace({'Yes': 1, 'No': 0}, inplace=True)
    pred['FrequentFlyer'].replace({'Yes': 1, 'No': 0}, inplace=True)
    pred['EverTravelledAbroad'].replace({'Yes': 1, 'No': 0}, inplace=True)
    pred['ChronicDiseases'].replace({'Yes': 1, 'No': 0}, inplace=True)

    if list(pred['Employment Type'])[0] == 'Government Sector':
        pred['Employment Type_Government Sector'] = True
        pred['Employment Type_Private Sector/Self Employed'] = False
    else:
        pred['Employment Type_Government Sector'] = False
        pred['Employment Type_Private Sector/Self Employed'] = True
    p_input = pred.drop(columns=['Employment Type'])
    return p_input

if st.button('Predict Buying'):
    input_df=pd.DataFrame({'Age':[Age],'Employment Type':[Employment_Type],'GraduateOrNot':[GraduateOrNot],
                           'AnnualIncome':[AnnualIncome],'FamilyMembers':[FamilyMembers],
                           'ChronicDiseases':[ChronicDiseases],'FrequentFlyer':[FrequentFlyer],
                           'EverTravelledAbroad':[EverTravelledAbroad]})
    # st.table(input_df)
    df=pre_processing(input_df)
    # st.table(df)
    output=rfc.predict(df)
    if output==1:
        st.header("YES, this person will buy Travel Insurance")
    else:
        st.header("NO, this person will not buy Travel Insurance")
