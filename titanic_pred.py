# -->Importing libraries<--
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly as px
import seaborn as sns; sns.set()
from st_on_hover_tabs import on_hover_tabs


#Imported models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import base64
import streamlit as st 
import streamlit_extras as ste
import streamlit_option_menu as sto

#Importing the dataset
dataset = pd.read_csv('titanic-passengers.csv', delimiter= ';', index_col='PassengerId')


#Initializing the page and setting a background
st.set_page_config(page_title = "Titanic Survival Prediction",
                    page_icon ='🌊',
                    layout= 'centered')

st.markdown('<style>' + open('./style.css').read() + '<style>', unsafe_allow_html= True)

with st.sidebar:
    tabs = on_hover_tabs(tabName= ['Home', 'EDA','Data Cleaning','Model', 'Predict','Visualization', 'Conclusion', 'About Us', 'Contact'], 
                         iconName = ['house-fill','publish','hub','api','tsunami','equalizer','dashboard','handshake', 'person'], default_choice= 0)
    


# # STEP 1: Store pages inside session_state
if 'page' not in st.session_state:
    st.session_state['page'] = 0

def next_page():
    st.session_state['page'] += 1

def previous_page():
    st.session_state['page'] -= 1
    

if tabs == 'Home':
    st.title('Home')
    st.write(''' Welcome aboard to Titanic Explorer!

🚢 Ahoy there! Prepare to embark on a fascinating journey as we set sail through the historic voyage of the Titanic. Welcome to Titanic Explorer, your one-stop app for diving deep into the legendary Titanic dataset and uncovering intriguing insights.

🏠 Home Segment: Let's start at home! This is your launchpad to navigate through all the incredible features of our app. Here, we'll give you a brief overview of what you can expect from each segment:

🗄️ Dataset: Delve into the heart of the Titanic dataset, where you can explore the vast collection of passenger information. Learn about their age, class, gender, and more, and understand how these factors influenced their fate during that fateful journey.

🔍 EDA (Exploratory Data Analysis): Uncover the hidden stories within the dataset through our comprehensive analysis. We've created interactive visualizations that provide a clear picture of trends, patterns, and correlations that existed on that ill-fated voyage.

🔮 App for Prediction: Ever wondered if you were aboard the Titanic, what would be your chances of survival? Our prediction section will predict your survival probability based on specific inputs. Explore various scenarios and see if you would've made it to safety.

📏 Evaluation Metric: Curious about how we measure the accuracy of our predictions? Our evaluation metric section will enlighten you on the methodologies and criteria we employ to ensure the precision and reliability of our predictions.

📊 Graphs: Visuals speak louder than words! In this segment, feast your eyes on a plethora of engaging charts and graphs that present a dynamic visualization of the Titanic dataset, making complex data more accessible and enjoyable.

👤 About Us: Get to know the passionate team behind Titanic Explorer. We are a dedicated group of data enthusiasts who are eager to share our knowledge and expertise with you.

📞 Contact: We value your feedback and inquiries. This section enables you to get in touch with us easily. Let us know what you think, and we'll be thrilled to assist you with any queries you may have.

🤝 At Titanic Explorer, we believe that learning can be both informative and enjoyable. Our user-friendly yet professional approach aims to make your experience seamless and engaging, regardless of your expertise level. So, whether you're an aspiring data scientist, history enthusiast, or simply curious about the Titanic, our app is tailor-made for you!

⚓ Get ready to unravel the captivating stories hidden within the Titanic dataset. Come aboard and explore the past like never before with Titanic Explorer. Let's navigate history together! ⚓

             
             ''')
    
if tabs == 'EDA':
    if st.session_state['page'] == 0:
        st.title("Exploratory Data Analysis")
        st.subheader('Dataset')
        st.divider()
        st.dataframe(dataset, use_container_width = True, height = 725)
        st.divider()
        st.button("Data Head -->", on_click = next_page)
    elif st.session_state['page'] == 1:
            head = dataset.head()
            st.title("Data Head")
            st.write("")
            st.dataframe(head, use_container_width = True, height = 225)
            st.divider()
               
            # Creating columns for navigating to next page and previous page
            col1, col2 = st.columns([1, 6])
            with col1:
                st.button("👈 Dataset", on_click = previous_page)

            with col2:
                st.button("Data Tail -->", on_click = next_page)
            
    elif st.session_state['page'] == 2:
            tail = dataset.tail()
            st.subheader("Data Tail")
            st.write("Finding the tail of our dataset means we look at the bottom 5 values of our dataset. This is used in exploratory data analysis as a way to share insight on large datasets, what is happening at the extreme end of our data. In pandas, this is accomplished using the code below. ")
            # Using Column Design
            col3, col4 = st.columns([1, 2])
            with col3:
                # Using Expander Design
                with st.expander("Code"):
                    st.code(
                        """
                        import pandas as pd
                
                        dataset = pd.read_csv("file_directory")
                        tail = dataset.tail()
                        """)
                with col4:
                    st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to finf the tail of the dataset.")
                    st.dataframe(tail, use_container_width = True, height = 225)
            col5, col6 = st.columns([1, 6])
            with col5:
                st.button("👈 Data Head", on_click = previous_page)

            with col6:
                st.button("Correlation Matrix -->", on_click = next_page)
                
    elif st.session_state['page'] == 3:
        
            correlation_matrix = dataset.corr()
            correlation_matrix = correlation_matrix.iloc[1:, :]
            st.subheader("Data Correlation Matrix")
            st.write("")
            st.dataframe(correlation_matrix, use_container_width = True, height = 750)
            col7, col8 = st.columns([1, 6])
            with col7:
                st.button("👈 Data Tail", on_click = previous_page)

            with col8:
                st.button("Null Count(Columns) -->", on_click = next_page)
                
    elif st.session_state['page'] == 4:
            check_null = dataset.isnull().sum()
            check_null = pd.Series(check_null, name = "Null_Value_Count")
            total_null = dataset.isnull().sum().sum()
            
            st.subheader("Null Value Count (Columns)")
            st.write("")
            st.dataframe(check_null, width = 250, height = 250)
            col9, col10 = st.columns([1, 6])
            with col9:
                st.button("👈 Null Count(Columns)", on_click = previous_page)

            with col10:
                st.button("Unique Count -->", on_click = next_page)
    elif st.session_state['page'] == 5:
            distinct_count = dataset.nunique()
            distinct_count = pd.Series(distinct_count, name = "Unique_Value_Count")
            st.subheader("Unique Values (Columns)")
            st.write("")
            st.dataframe(distinct_count, width = 250, height = 250)
            col11, col12 = st.columns([1, 6])
            with col11:
                st.button("👈 Unique Count", on_click = previous_page)

            with col12:
                st.button("Descriptive Statistics -->", on_click = next_page)
    elif st.session_state['page'] == 6:
            descriptive_stats = dataset.describe()
            st.subheader("Data Descriptive Statistics")
            st.write("")
            st.dataframe(descriptive_stats, height = 325)
            col13, col14 = st.columns([1, 6])
            with col13:
                st.button("👈 Unique Count", on_click = previous_page)

            with col14:
                pass
            
        
            
            
            
            
            
            
 

# if tabs == 'Data Cleaning':
    
# if tabs == 'Model':
    
# if tabs == 'Predict':
    
# if tabs == 'Visualization':
    
# if tabs == 'Conclusion':
    
# if tabs == 'About Us':
    
# if tabs == 'Contact':
    
    
    
    
#BACKGROUND
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-repeat: repeat;
        background-attachment: scroll;
    }}
    </style>
    """,
        unsafe_allow_html=True)


# Set a background image
background = add_bg_from_local('magicpattern-grid-pattern.png')


