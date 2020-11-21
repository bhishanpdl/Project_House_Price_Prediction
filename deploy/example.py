# Imports
import numpy as np
import pandas as pd
import os
import time

# internet
import urllib
import codecs
import base64

# special Imports
import shap
import pandas_profiling
import sweetviz as sv
import streamlit as st
import streamlit.components.v1 as stc
from streamlit_pandas_profiling import st_profile_report

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# local imports
import config
import util
from util import clean_data
from util import print_regr_eval

# modelling
import catboost as cb

# settings
st.set_option("deprecation.showfileUploaderEncoding", False)

__doc__ = """
Date   : Nov 16, 2020
Author : Bhishan Poudel
Purpose: Interactive report of the project

Command:
streamlit run app.py

"""

#============================ Read local markdown file
page = codecs.open('about.md','r').read()
# st.markdown(page)


#========================== read github markdown
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

# readme_text = st.markdown(get_file_content_as_string("instructions.md"))

#========================== file upload
upload = False
if upload:
    uploaded_file_buffer = st.file_uploader("")
    df_test = pd.read_csv(uploaded_file_buffer)
    df_test = df_test.head(100)
    st.text(f"Shape of Test Data: {df_test.shape}")
    st.dataframe(df_test.head())

#============================== Manually enter data
age = st.slider("Select Age",16,60)
education_level = st.number_input("Education Level (low2High) [1,4]",1,4)

salary = st.number_input("Salary")

reg = {"Non_Religious":0,"Religious":1}
choice = st.radio("Religious or not",tuple(reg.keys()))
religious = reg[choice]

working = {"Yes":1,"No":0}
choice= st.radio("Are You Currently Working",tuple(working.keys()))
isworking = working[choice]
occupation = st.selectbox("Your Occupation",['Data Scientist','Programmer','Doctor','Businessman'])

favorite_places = st.multiselect("Your Favorite Places",("London","New York","Kathmandu","Kiev","Berlin","Tokyo"))

data = {'age': [age],
        'education_level': [education_level],
        'Salary': [salary],
        'religious': [religious],
        'isworking': [isworking],
        'occupation': [occupation],
        'favorite_places': [favorite_places]
        }

df = pd.DataFrame(data)
st.dataframe(df)

st.header("Entering Texts")
name = st.text_input("Enter Name","Tom Cruise")
if st.button('Submit'):
    result_name = name.title()
    st.success("You entered " + result_name)
else:
    st.write("Press the above button..")

c_text = st.text_area("Enter Text","Hello")
if st.button('Analyze'):
    result_text = c_text.title()
    st.success("You entered " + result_text)
else:
    st.write("Press the above button..")

#======================= Warnings
st.header("Warnings and Errors")
st.info("This is an info alert ")
st.warning("This is a warning ")
st.error("This shows an error ")

#======================== Date time
st.header("Date time")
import datetime,time
today = st.date_input("Today is",datetime.datetime.now())
t = st.time_input("The time now is",datetime.time())

#======================== Json and code
st.header("JSON and Code")
# Display JSON
st.text("Display JSON")
st.json({'name':'hello','age':34})

# Display Raw Code
st.text("Display Raw Code Using st.code")
st.code("import numpy as np") # st.code('hello.py')


st.text("Display Raw Code Using st.echo")
with st.echo():
	# This will also be shown
	import pandas as pd

	df = pd.DataFrame()

#================================= Spinner
st.header("Spinner and progress")
with st.spinner("Waiting .."):
	time.sleep(3)
st.success("Finished!")

# import time
# my_bar = st.progress(0)
# for p  in range(3):
# 	my_bar.progress(p +1)