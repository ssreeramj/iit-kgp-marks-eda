import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.precision", 2)

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

st.title('Semester 6 marks analysis')
st.header('Lets have a look at the marks scored by all students')
# st.empty()

clean_df = load_data()
student_df = clean_df.iloc[:, :5].set_index('RollNo').sort_values(by=['SGPA', 'CGPA'], ascending=False)

# students marks filter
st.sidebar.header('Students marks analysis')

# cg filter
cg_slider = st.sidebar.slider("CGPA Filter", 0.0, 10.0, (0.0, 10.0), 0.1, format='%f')

# sg filter
sg_slider = st.sidebar.slider("SGPA Filter", 0.0, 10.0, (0.0, 10.0), 0.1, format='%f')


# dep filter
dep_options = student_df['Dept'].unique()
dep_options = np.insert(dep_options, 0, 'ALL') 
dep_filter = st.sidebar.selectbox('Department', dep_options)

filter_student_df = student_df.query(
    f'SGPA <= {str(sg_slider[1])} & SGPA >= {str(sg_slider[0])} \
        & CGPA <= {str(cg_slider[1])} & CGPA >= {str(cg_slider[0])}'
)

if dep_filter != 'ALL':
    filter_student_df.query(f'Dept == "{dep_filter}"', inplace=True)

# is_name_filter = st.checkbox(('Find a name'))
name_filter = st.sidebar.multiselect('Name', student_df['Name'].values)

if len(name_filter) >= 1:
    filter_student_df = filter_student_df.loc[filter_student_df['Name'].isin(name_filter)]

st.dataframe(filter_student_df)

if name_filter:
    student_name = name_filter[-1]
    student_subjects = clean_df.loc[clean_df['Name'] == student_name]
    
    student_sub_df = student_subjects[['Subject1', 'Grade1']].rename(columns={ 'Subject1': 'Subject', 'Grade1': 'Grade' })

    for i in range(2, 12):
        student_sub_df = pd.concat([student_sub_df, 
               student_subjects[[f'Subject{i}', f'Grade{i}']].rename(
                   columns={ f'Subject{i}': 'Subject', f'Grade{i}': 'Grade' })], 
                               ignore_index=True)    
    
    student_sub_df = student_sub_df.dropna().drop_duplicates()
    
    fig = px.strip(student_sub_df, x='Subject', y='Grade', title=f'{student_name}', 
                   category_orders={ 'Grade': ['EX', 'A', 'B', 'C', 'D', 'P', 'F', 'X', 'Y']}, height=450, width=600)
    
    st.plotly_chart(fig, use_container_width=True)
    # st.write(student_name)

st.write('We can have a look at various statistical properties of SGPA in each department.\
     We can have a look at the histogram plot of SGPA and CGPA in a specific department by clicking it in the table')


dept_df = pd.DataFrame(clean_df.groupby('Dept')['SGPA'].agg(['max', 'min', 'mean', 'median'])).reset_index()

if dep_filter != 'ALL':
    hist_fig = px.histogram(clean_df.loc[clean_df.Dept == dep_filter], x=['SGPA', 'CGPA'], barmode="overlay",
                            title=f'Marks Distribution among students in {dep_filter}', height=450, width=550)

    st.plotly_chart(hist_fig, use_container_width=True)

st.dataframe(dept_df, width=500,)

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

subject_df = df_marks[['Subject1', 'Grade1']].rename(columns={ 'Subject1': 'Subject', 'Grade1': 'Grade' })

for i in range(2, 12):
    subject_df = pd.concat([subject_df, 
           df_marks[[f'Subject{i}', f'Grade{i}']].rename(columns={ f'Subject{i}': 'Subject', f'Grade{i}': 'Grade' })], 
                           ignore_index=True)
    
subject_df = subject_df.drop(subject_df.loc[subject_df['Subject'] == 0].index, axis=0).reset_index(drop=True)

subject_dist = subject_df.groupby('Subject')['Grade'].agg(['mean', 'count']).sort_values(['mean', 'count'], ascending=False).reset_index()
subject_dist = subject_dist.loc[subject_dist['count'] >= 10]

st.dataframe(subject_dist)

subject_filter = st.text_input('Subject name', value='NA31002')

temp_subject = subject_df[subject_df['Subject'] == subject_filter].groupby(
                        ['Grade']).count().reset_index(level=[0])
    
temp_subject['Grade'] = temp_subject['Grade'].replace(
    {10: 'EX', 9: 'A', 8: 'B', 7: 'C', 6: 'D', 5: 'P', 0: 'F'})

temp_subject = temp_subject.rename(columns={'Subject': 'Count'})

pie_fig = px.pie(temp_subject, values='Count', names='Grade', title=f'Grade distribution for {subject_filter}', height=450, width=550)
pie_fig.update_traces(textinfo='percent+label')

st.plotly_chart(pie_fig)