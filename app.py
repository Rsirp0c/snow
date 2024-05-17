import streamlit as st
import pandas as pd
import os
import replicate
# import cohere
# from langchain_cohere import ChatCohere
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
import plotly.express as px
from transformers import AutoTokenizer

st.title("Data Exploration with LLM")

# if "cohere_api" not in st.session_state:
#     st.session_state.cohere_api = st.secrets["cohere_api"]
    
# co = cohere.Client(st.session_state.cohere_api)

with st.sidebar:
    st.title('Snowflake Arctic Setting')
    if st.secrets and 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your Replicate API token.', icon='⚠️')
            st.markdown("**Don't have an API token?** Head over to [Replicate](https://replicate.com) to sign up for one.")

    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    st.subheader("Adjust model parameters")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.3, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

def get_tokenizer():
    return AutoTokenizer.from_pretrained(
    'Snowflake/snowflake-arctic-embed-l',
    trust_remote_code=True)

def get_num_tokens(prompt):
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

# TODO: prompt engineering
def get_response(prompt):
    if get_num_tokens(prompt) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.stop()
    response = ''
    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                           input={"prompt": prompt,
                                  "prompt_template": "<|im_start|>system\nYou're a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n\n<|im_start|>assistant\n",
                                  "temperature": temperature,
                                  "top_p": top_p,
                                  }):
        response += str(event)
    return response
    # return co.client(message = prompt)

# TODO: new function to generate suggestions by Artic?
def get_suggestions(header):
    prompt = '''Generate a list of prompts for creating data visualizations from CSV files.
                            Each prompt should specify the type of data to be visualized, the key metrics 
                            or attributes involved, and the recommended types of visualizations to use. 
                            It may include diverse categories such as {header}. Return a list(with no variable name) of string in python'''
    suggestions = ''
    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                        input={"prompt": prompt,
                                "prompt_template": "<|im_start|>system\nYou're a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n\n<|im_start|>assistant\n",
                                "temperature": temperature,
                                "top_p": top_p,
                                }):
        print(str(event))
        suggestions += str(event)
    return suggestions

def render_suggestions(header):
    def set_query(query):
        st.session_state.suggestion = query
    # suggestions = [
        
    #     "Compare Electric Vehicle Types in the data",
    #     "Market Trends by Make and Model",
    #     "Electric Range of different models and makes",
    # ]
    local_vars = {}
    
    # Get suggestions and ensure they are in list format
    suggestions = get_suggestions(header)
    prompts = suggestions.replace('\n', ' ').replace('  ', ' ').split('", "')
    prompts = [prompt.strip().strip('["').strip('"]') for prompt in prompts][:2]

    columns = st.columns(2)
    for i, column in enumerate(columns):
        with column:
            st.button(prompts[i], on_click=set_query, args=[suggestions[i]], key=f"prompts_{i}")


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

def generate_visualization_code(query, header):
    question =  f"""
                here is the code I have already: 

                <<<
                import streamlit as st
                import pandas as pd
                import plotly.express as px


                df = pd.DataFrame(data)
                <<<

                the schema of df is as follows, please use the exact column names to refer to the data:
               {header}
                
                use streamlit and plotly to visualize the data based on this insight:
                {query}

                only return the code I need to add upon my code, and use the variable df to refer to the dataframe
                """
    print("question: ",question)
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

data = pd.read_csv('Electric_Vehicle_Population_Data.csv')

df = pd.DataFrame(data)
header = ', '.join(data.columns.tolist())
st.dataframe(df.head(10))

render_suggestions(header)
render_query()

if not user_query:
    st.stop()

viz_code = generate_visualization_code(user_query, header)
print(viz_code)

start = viz_code.find("python") + len("python")
end = viz_code.find("```", start)

extracted_code = viz_code[start:end].strip()

f"{viz_code}"

llm_code = f"{extracted_code}"
llm_code = llm_code.strip('```python').strip('```').strip()
exec(llm_code)
