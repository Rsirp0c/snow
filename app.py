import streamlit as st
import pandas as pd
import cohere
# from langchain_cohere import ChatCohere
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
import plotly.express as px

st.title("Data Exploration with LLM")

if "cohere_api" not in st.session_state:
    st.session_state.cohere_api = st.secrets["cohere_api"]
    
co = cohere.Client(st.session_state.cohere_api)

def get_response(prompt):
    response = co.chat(message = prompt)
    return response

def render_suggestions():
    def set_query(query):
        st.session_state.suggestion = query

    suggestions = [
        
        "Compare Electric Vehicle Types in the data",
        "Market Trends by Make and Model",
        "Electric Range of different models and makes",
    ]

    columns = st.columns(len(suggestions))
    for i, column in enumerate(columns):
        with column:
            st.button(suggestions[i], on_click=set_query, args=[suggestions[i]])


def render_query():
    st.text_input(
        "query",
        placeholder="Enter what you want to explore in the data...",
        key="user_query",
        label_visibility="collapsed",
    )


def get_query():
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = None
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    user_query = st.session_state.suggestion or st.session_state.user_query
    st.session_state.suggestion = None
    st.session_state.user_query = ""
    return user_query

def generate_visualization_code(query):
    question =  f"""
                here is the code I have already: 

                <<<
                import streamlit as st
                import pandas as pd
                import plotly.express as px

                data = pd.read_csv('Electric_Vehicle_Population_Data.csv')

                df = pd.DataFrame(data)
                <<<

                the schema of the data is as follows:
                VIN (1-10),County,City,State,Postal Code,Model Year,Make,Model,Electric Vehicle Type,Clean Alternative Fuel Vehicle (CAFV) Eligibility,Electric Range,Base MSRP,Legislative District,DOL Vehicle ID,Vehicle Location,Electric Utility,2020 Census Tract
                
                write the code to visualize the data based on this insight:
                {query}

                only return the code I need to add upon my code
                do not include any texts in the response, I need to execute the response as code
                """
    code = get_response(question)

    return code


user_query = get_query()
if not user_query:
    st.info(
        "Upload a csv file and use llm to do data exploration. Type a query or pick one suggestions:"
    )
col1,col2 = st.columns([1,1])
with col1:
    st.file_uploader("Upload CSV file", type="csv",label_visibility="collapsed")
with col2:
    with st.container(border=True):
        '''
        If you haven't uploaded a file, use the provided sample data.
        [Electric Vehicle Population Data](https://catalog.data.gov/dataset/electric-vehicle-population-data)
        '''
render_suggestions()
render_query()

data = pd.read_csv('Electric_Vehicle_Population_Data.csv')

df = pd.DataFrame(data)

if not user_query:
    st.stop()

viz_code = generate_visualization_code(user_query)


f"{viz_code.text}"
llm_code = f"{viz_code.text}"
llm_code = llm_code.strip('```python').strip('```').strip()
exec(llm_code)
