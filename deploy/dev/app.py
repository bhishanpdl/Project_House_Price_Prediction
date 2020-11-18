import streamlit as st
import os
import glob
import pandas as pd
import shutil
from PIL import Image
import base64
import io

# data viz
import matplotlib
matplotlib.use("Agg")  # To Prevent Errors
import matplotlib.pyplot as plt
import seaborn as sns

# settings
st.set_option('deprecation.showfileUploaderEncoding', False)

__doc__ = """
Date   : Nov 16, 2020
Author : Bhishan Poudel
Purpose: Interactive report of the project

Command:
streamlit run app.py


"""

# decorator function for logging
def st_log(func):
    def log_func(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time() - start
        st.text("Log: the function `%s` tooks %0.4f seconds" % (func.__name__, end))
        return res

    return log_func


def main():
    """Main function """

    html = """
	<div style="background-color:tomato;"><p style="color:white;font-size:30px;"> King County House Price Prediction</p></div>
	"""
    st.markdown(html, unsafe_allow_html=True)

    html = """<marquee style='width: 30%; color: blue;'><b> Author: Bhishan Poudel</b></marquee>"""
    st.markdown(html, unsafe_allow_html=True)

    st.subheader("Purpose: Predict the price of house from given features")

    def file_selector(folder_path="../data/raw"):
        filenames = os.listdir(folder_path)
        filenames = [i for i in filenames if i.endswith(".csv")]
        selected_filename = st.selectbox("Select a file", filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    st.write("You selected `%s`" % filename)
    df = pd.read_csv(filename)

    # Show Dataset
    if st.checkbox("Show DataSet Header"):
        number = int(st.slider("Number of Rows to View"))
        if number <= 5:
            number = 5
        st.dataframe(df.head(int(number)))
    # Show Column Names
    if st.checkbox("Columns Names"):
        st.write(df.columns.tolist())

    # Show Shape of Dataset
    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)

    # Show Columns By Selection
    selected_col = 'bedrooms'
    if st.checkbox("Select Columns To Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select", all_columns)
        selected_col = selected_columns[0]
        new_df = df[selected_columns]
        st.dataframe(new_df.head())

    # Value Counts
    if st.checkbox("Value Counts for " + str(selected_col)):
        st.write(df[selected_col].value_counts())

    # Summary
    if st.checkbox("Summary"):
        st.write(df.describe())

    st.subheader("Data Visualization")
    # Show Correlation Plots
    # Seaborn Plot
    if st.checkbox("Correlation Plot"):
        st.write(sns.heatmap(df.corr(), annot=False))
        st.pyplot()

    # Counts Plots
    if st.checkbox("Plot of Value Counts"):
        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox(
            "Select Primary Column To Group By", all_columns_names
        )
        selected_column_names = st.multiselect("Select Columns", all_columns_names)
        if st.button("Plot"):
            st.text(
                "Generating Plot for: {} and {}".format(
                    primary_col, selected_column_names
                )
            )
            if selected_column_names:
                vc_plot = df.groupby(primary_col)[selected_column_names].count()
            else:
                vc_plot = df.iloc[:, -1].value_counts()
            st.write(vc_plot.plot(kind="bar"))
            st.pyplot()

    st.subheader("Our Features and Target")

    if st.checkbox("Show Features"):
        all_features = df.iloc[:, 0:-1]
        st.text("Features Names:: {}".format(all_features.columns[0:-1].to_numpy()))
        st.dataframe(all_features.head())

    # Download a file
    df_test = None
    df_test_sample = pd.read_csv('../data/raw/test.csv').sample(10)
    if st.checkbox("Download Sample Test data"):
        csv = df_test_sample.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (Downloaded file name is download, rename it to something.csv)'
        st.markdown(href, unsafe_allow_html=True)

    if st.checkbox("Upload Test data"):
        uploaded_file_buffer = st.file_uploader("Upload")
        # uploaded_file_text = io.TextIOWrapper(uploaded_file_buffer)
        df_test = pd.read_csv(uploaded_file_buffer)
        st.text(df_test.head())

if __name__ == "__main__":
    main()
