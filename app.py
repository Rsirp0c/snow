import streamlit as st
import pandas as pd
import os
import time
import replicate
# import cohere
# from langchain_cohere import ChatCohere
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
import plotly
import plotly.express as px
import seaborn
from transformers import AutoTokenizer

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
    prompt = f'''
            Generate 3 prompts for creating data visualization code based on the data schema: {header}. 
            Only return a Python String list of generated prompts as the following format: ['prompt1', 'prompt2', 'prompt3']. 
            The space betwen each element of list shuld be one.
            '''
    suggestions = ''
    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                        input={"prompt": prompt,
                                "prompt_template": "<|im_start|>system\nYou're a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n\n<|im_start|>assistant\n",
                                "temperature": temperature,
                                "top_p": top_p,
                                }):
        suggestions += str(event)
    return suggestions

def render_suggestions(header):

    def set_query(query):
        st.session_state.suggestion = query

    # Get suggestions and ensure they are in list format
    if st.session_state.render:
        suggestions = get_suggestions(header)
        prompts = suggestions.replace('\n', ' ').replace('  ', ' ').replace('\', \'', '", "').replace('","', '", "').replace('\',\'', '", "').split('", "')
        prompts = [prompt.strip().strip('["').strip('"]').strip('"').strip('\'') for prompt in prompts][:3]
        st.session_state.prompts = prompts 
        st.session_state.render = False
    else:
        prompts = st.session_state.prompts

    with st.status("Rendering suggestions...", expanded=True) as status:
        "#### Suggestions by Snowflake Arctic 💡"

        # columns = st.columns(3)
        # for i, column in enumerate(columns):
        #     with column:
        #         st.button(prompts[i], on_click=set_query, args=[suggestions[i]], key=f"prompts_{i}")
    
        for i in range(3):
            st.button(prompts[i],on_click=set_query, args=[prompts[i]], key=f"prompts_{i}", use_container_width=True)
            status.update(label="Complete!", state="complete", expanded=True)
        if st.button("Refresh suggestions"):
            st.session_state.render = True
            st.rerun() 
    

def render_query():
    st.text_input(
        "query",
        placeholder="Enter what you want to explore in the data...",
        key="user_query",
        label_visibility="collapsed",
    )


def get_query():
    # Initialize session state variables
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = None
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "prompts" not in st.session_state:
        st.session_state.prompts = None
    if "render" not in st.session_state:
        st.session_state.render = True

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
                import plotly

                # df is the dataframe,run operations on it
                >>>

                the schema of df is as follows, please use the exact column names to refer to the data:
               {header}
                
                use streamlit and plotly to visualize the data based on this insigh, don't use "fig.show()":
                {query}

                only return the code I need to add upon my code, and use the variable df to refer to the dataframe
                """
    print("question: ",question)
    code = get_response(question)

    return code


@st.cache_data(experimental_allow_widgets = True)
def get_data():
    # st.cache_data.clear()
    col1,col2 = st.columns([1.1,1])
    with col1:
        raw_data = st.file_uploader("Upload CSV file", type="csv",label_visibility="collapsed",key="unique_key_1")
        '''
        [Click here](https://github.com/laxmimerit/All-CSV-ML-Data-Files-Download/tree/master) to view more datasets.
        '''
        # Using a [NBA dataset](https://github.com/laxmimerit/All-CSV-ML-Data-Files-Download/blob/master/nba.csv) as default.
    with col2:
        # with st.container(border=True):
        default_data = st.radio("Select a dataset", ["nba.csv", "snowflake.csv", "titanic.csv"],index=0, key="unique_key_2")
        if raw_data is None:
            raw_data = default_data
    
    st.session_state.render = True  
    data = pd.read_csv(raw_data)
    header = ', '.join(data.columns.tolist())
    return data, header

####################

st.header("Data Exploration with :blue[Snowflake Arctic]")

user_query = get_query()
if not user_query:
    st.info(
        "Upload a csv file and use llm to do data exploration. Type a query or pick one from suggestions:"
    )
    
# start_time = time.time()
df,header = get_data()
# st.write(time.time()-start_time)

st.dataframe(df,use_container_width=True, hide_index=True, height=248)

render_suggestions(header)
render_query()

# with st.sidebar:
#     for element in st.session_state:
#         st.write(f"{element}: {st.session_state[element]}")

if not user_query:
    st.stop()

code_box = st.container().empty()
with code_box.status("fetching visualization code...", expanded=False):
    viz_code = generate_visualization_code(user_query, header)
    # print(viz_code)
    start = viz_code.find("python") + len("python")
    end = viz_code.find("```", start)
    extracted_code = viz_code[start:end].strip()
    f"{viz_code}"


llm_code = extracted_code.strip('```python').strip('```').strip()
exec(llm_code)

