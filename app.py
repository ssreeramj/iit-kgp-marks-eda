import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

import warnings
warnings.filterwarnings('ignore')

CSV_PATH = '17f.csv'

@st.cache
def load_data():
    col_names = [
        'Name', 'RollNo', 'Score', 'Subject1', 'Grade1', 'Subject2', 'Grade2', 'Subject3', 'Grade3', 'Subject4', 'Grade4',
        'Subject5', 'Grade5', 'Subject6', 'Grade6', 'Subject7', 'Grade7', 'Subject8', 'Grade8', 'Subject9', 'Grade9',
        'Subject10', 'Grade10', 'Subject11', 'Grade11',
    ]
    df = pd.read_csv(CSV_PATH, header=None, names=col_names)

    df[['NCGPA', 'CGPA', 'SGPA']] = df.Score.str.split(" - ", expand=True)
    df['Dept'] = df.RollNo.str.slice(2, 4,)
    df.drop(['Score', 'NCGPA'], axis=1, inplace=True)

    df[['SGPA', 'CGPA']] = df[['SGPA', 'CGPA']].apply(pd.to_numeric) 

    df = df.drop(index=df.loc[df['SGPA'] == 0].index, axis=0).reset_index(drop=True)

    cols = df.columns.tolist()
    cols = cols[:2] + ['Dept', 'SGPA', 'CGPA'] + cols[2:-3]
    df = df[cols]

    return df

def plot_subject_grades(selected_names):
    student_name = selected_names[-1]
    student_subjects = clean_df.loc[clean_df['Name'] == student_name]
    
    student_sub_df = student_subjects[['Subject1', 'Grade1']].rename(columns={ 'Subject1': 'Subject', 'Grade1': 'Grade' })

    for i in range(2, 12):
        student_sub_df = pd.concat([student_sub_df, 
               student_subjects[[f'Subject{i}', f'Grade{i}']].rename(
                   columns={ f'Subject{i}': 'Subject', f'Grade{i}': 'Grade' })], 
                               ignore_index=True)    
    
    student_sub_df = student_sub_df.dropna().drop_duplicates()
    
    strip_fig = px.strip(student_sub_df, x='Subject', y='Grade', 
                   category_orders={ 'Grade': ['EX', 'A', 'B', 'C', 'D', 'P', 'F', 'X', 'Y']})

    return strip_fig

def get_subject_df(marks_data):
    sub_df = marks_data[['Subject1', 'Grade1']].rename(columns={ 'Subject1': 'Subject', 'Grade1': 'Grade' })

    for i in range(2, 12):

        sub_df = pd.concat([sub_df, 
            df_marks[[f'Subject{i}', f'Grade{i}']].rename(columns={ f'Subject{i}': 'Subject', f'Grade{i}': 'Grade' })], 
                            ignore_index=True)
        
    sub_df = sub_df.drop(sub_df.loc[sub_df['Subject'] == 0].index, axis=0).reset_index(drop=True)

    return sub_df

def get_sub_dist_df(marks_data, min_students):
    sub_df = get_subject_df(marks_data)

    subject_dist = sub_df.groupby('Subject')['Grade'].agg(['mean', 'count']).sort_values(['mean', 'count'], ascending=False).reset_index()
    subject_dist = subject_dist.loc[subject_dist['count'] >= min_students]

    return subject_dist

def plot_subject_pie(marks_data, subject_name):
    df = get_subject_df(marks_data)

    temp_subject = df[df['Subject'] == subject_name].groupby(
                            ['Grade']).count().reset_index(level=[0])
        
    temp_subject['Grade'] = temp_subject['Grade'].replace(
        {10: 'EX', 9: 'A', 8: 'B', 7: 'C', 6: 'D', 5: 'P', 0: 'F'})

    temp_subject = temp_subject.rename(columns={'Subject': 'Count'})

    pie_fig = px.pie(temp_subject, values='Count', names='Grade')
    pie_fig.update_traces(textinfo='percent+label')

    return pie_fig

if __name__ == '__main__':

    st.title('IIT-KGP Marks EDA :sunglasses:')

    st.write('We have the marks of 3rd year students of Spring Semester 2019-2020. \
        This is a simple app which helps us do some analysis of the marks obtained by students, average marks in a department \
            and see how tough is a particular course')

    st.info('All the tables can be sorted by any column by clicking on the column name')

    st.header('All students SGPA and CGPA :fire:')

    st.write('We can have a look at the score obtained by all students sorted by SGPA column by default')

    # st.empty()

    clean_df = load_data()
    student_df = clean_df.iloc[:, :5].set_index('RollNo').sort_values(by=['SGPA', 'CGPA'], ascending=False)

    st.sidebar.title('Filters')
    st.sidebar.markdown('##### Can be used to filter out the table in a specific CGPA/SGPA range')
    cg_slider = st.sidebar.slider("CGPA Filter", 0.0, 10.0, (0.0, 10.0), 0.1, format='%f')
    sg_slider = st.sidebar.slider("SGPA Filter", 0.0, 10.0, (0.0, 10.0), 0.1, format='%f')


    st.sidebar.markdown('##### By selecting a department, we see the histogram of the CGPA and SGPA')
    # dep filter
    dep_options = sorted(student_df['Dept'].unique().tolist())
    dep_options = ['ALL'] + dep_options 
    dep_filter = st.sidebar.selectbox('Department', dep_options)

    filter_student_df = student_df.query(
        f'SGPA <= {str(sg_slider[1])} & SGPA >= {str(sg_slider[0])} \
            & CGPA <= {str(cg_slider[1])} & CGPA >= {str(cg_slider[0])}'
    )

    if dep_filter != 'ALL':
        filter_student_df.query(f'Dept == "{dep_filter}"', inplace=True)

    # is_name_filter = st.checkbox(('Find a name'))
    st.sidebar.markdown('##### We can look into the grades of a particular student by searching here')
    name_filter = st.sidebar.multiselect('Name', student_df['Name'].values)

    if name_filter:
        filter_student_df = filter_student_df.loc[filter_student_df['Name'].isin(name_filter)]

        subject_figure = plot_subject_grades(name_filter)
        

    st.dataframe(filter_student_df.reset_index().style.format({'SGPA': '{:.2f}', 'CGPA': '{:.2f}'}))

    if name_filter:
        st.subheader(f'{name_filter[-1]}')

        st.plotly_chart(subject_figure, use_container_width=True)

    st.header('Departments :mortar_board:')

    dept_df = pd.DataFrame(clean_df.groupby('Dept')['SGPA'].agg(['max', 'min', 'mean', 'median'])).reset_index()

    if dep_filter != 'ALL':
        st.subheader(f'SGPA and CGPA distribution among students in {dep_filter} :bar_chart:')
        
        hist_fig = px.histogram(clean_df.loc[clean_df.Dept == dep_filter], x=['SGPA', 'CGPA'], barmode="overlay")

        st.plotly_chart(hist_fig, use_container_width=True)

    st.write('We can have a look at various statistical properties of SGPA in each department.\
        We can have a look at the histogram plot of SGPA and CGPA in a specific department by selecting a \
            a particular department from the sidebar')

    st.dataframe(dept_df.style.format({'min': '{:.2f}', 'max': '{:.2f}', 'mean': '{:.2f}', 'median': '{:.2f}'}))

    grade_col_names = [
        'Grade1', 'Grade2', 'Grade3', 'Grade4', 'Grade5', 'Grade6', 'Grade7', 'Grade8', 'Grade9', 'Grade10', 'Grade11'
    ]

    df_marks = clean_df.copy()
    df_marks[grade_col_names] = df_marks[grade_col_names].replace({
        'EX': 10,
        'A': 9,
        'B': 8,
        'C': 7,
        'D': 6,
        'P': 5,
        'X': 0,
        'F': 0,
        'Y': 0,
    })
    df_marks = df_marks.fillna(0)

    st.sidebar.markdown('##### We can filter courses where atleast these many students were enrolled')
    min_students = st.sidebar.slider('Min. students', 1, 210, value=10, step=1, format='%d')

    subject_dist = get_sub_dist_df(df_marks, min_students)

    st.header('Subjects :chart_with_upwards_trend:')

    st.write('We take a look at the average grades obtained by students in a subject. The mean score is \
        calculated by taking EX=10, A=9 and so on.')

    st.dataframe(subject_dist)

    subject_options = get_sub_dist_df(df_marks, 1)['Subject'].unique().tolist()

    st.sidebar.markdown('##### Select a subject to get the grade distribution')
    subject_filter = st.sidebar.multiselect('Subject', subject_options)

    if len(subject_filter) >= 1:
        pie_plot = plot_subject_pie(df_marks, subject_filter[-1])

        st.subheader(f'Grade Distribution for {subject_filter[-1]}')
        st.plotly_chart(pie_plot)