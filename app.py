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
import streamlit.components.v1 as components

with st.sidebar:
    st.title('Snowflake Arctic Setting')
    if os.path.isfile('.streamlit/secrets.toml'):
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
        # col1, col2 = st.columns([1.1, 1.06])
        # with col1:
        st.markdown("#### Suggestions by Snowflake Arctic 💡")
        # with col2:
        #     st.markdown("""
        #     <style>
        #         .streamlit-container {
        #             display: flex;
        #             align-items: flex-end:;  # This centers the logo vertically
        #             justify-content: flex-end;  # This centers the logo horizontally
        #         }
        #     </style>
        #     <svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 398 398" width="38" height="38">
        #     <defs>
        #         <style>
        #         .cls-1 {
        #             fill: #2bb5e9;
        #             fill-rule: evenodd;
        #             stroke-width: 0px;
        #         }
        #         </style>
        #     </defs>
        #     <path class="cls-1" d="M380.36,59.35h-3.6v4.2h3.6c1.8,0,2.4-.6,2.4-1.8.6-1.2-.6-2.4-2.4-2.4ZM372.55,55.74h7.81c4.2,0,7.21,2.4,7.21,6.01,0,2.4-1.2,3.6-3,4.81l3,4.2v.6h-4.2l-3-4.2h-3.6v4.2h-4.2v-15.62ZM394.17,64.15c0-8.41-6.01-15.02-14.42-15.02s-14.42,6.01-14.42,15.02c0,8.41,6.01,15.02,14.42,15.02,9.01,0,14.42-6.61,14.42-15.02ZM397.78,64.15c0,10.21-6.61,18.62-18.02,18.62s-18.02-8.41-18.02-18.62,6.61-18.62,18.02-18.62c11.41,0,18.02,8.41,18.02,18.62Z"/>
        #     <path class="cls-1" d="M367.08,174.47l-42.55,24.61,42.55,24.54c10.75,6.22,14.42,19.9,8.21,30.67-6.2,10.7-19.92,14.37-30.63,8.18l-76.21-44c-5.08-2.96-8.61-7.59-10.18-12.8-.74-2.4-1.09-4.87-1-7.31.04-1.76.3-3.53.79-5.29,1.52-5.48,5.1-10.38,10.39-13.46l76.21-43.97c10.71-6.19,24.43-2.51,30.63,8.22,6.22,10.72,2.54,24.41-8.21,30.61ZM326.8,293.75l-76.19-43.94c-4.09-2.38-8.63-3.3-12.99-2.93-11.62.83-20.75,10.53-20.75,22.34v87.98c0,12.4,10,22.45,22.42,22.45s22.45-10.05,22.45-22.45v-49.21l42.66,24.63c10.72,6.23,24.43,2.54,30.61-8.17,6.17-10.74,2.53-24.5-8.21-30.7ZM238.88,207.68l-31.67,31.63c-.91.93-2.65,1.69-3.98,1.69h-9.31c-1.28,0-3.06-.76-3.98-1.69l-31.66-31.63c-.91-.89-1.64-2.68-1.64-3.94v-9.33c0-1.29.73-3.09,1.64-3.99l31.66-31.63c.92-.93,2.7-1.66,3.98-1.66h9.31c1.29,0,3.07.73,3.98,1.66l31.67,31.63c.9.9,1.63,2.7,1.63,3.99v9.33c0,1.26-.73,3.05-1.63,3.94ZM213.56,198.9c0-1.28-.77-3.06-1.68-4l-9.17-9.13c-.9-.9-2.68-1.65-3.96-1.65h-.36c-1.28,0-3.05.75-3.94,1.65l-9.17,9.13c-.92.95-1.62,2.73-1.62,4v.36c0,1.26.7,3.03,1.62,3.95l9.17,9.16c.9.9,2.67,1.65,3.94,1.65h.36c1.28,0,3.06-.75,3.96-1.65l9.17-9.16c.9-.92,1.68-2.68,1.68-3.95v-.36ZM70.35,104.35l76.21,43.99c4.09,2.35,8.64,3.29,13.01,2.93,11.6-.86,20.75-10.58,20.75-22.37V40.91c0-12.37-10.06-22.41-22.42-22.41s-22.45,10.05-22.45,22.41v49.23l-42.7-24.66c-10.71-6.2-24.4-2.53-30.61,8.2-6.19,10.76-2.53,24.47,8.2,30.67ZM237.62,151.26c4.36.36,8.9-.57,12.99-2.93l76.19-43.99c10.74-6.2,14.38-19.91,8.21-30.67-6.18-10.72-19.89-14.4-30.61-8.2l-42.66,24.66v-49.23c0-12.37-10.03-22.41-22.45-22.41s-22.42,10.05-22.42,22.41v87.98c0,11.8,9.13,21.51,20.75,22.37ZM159.57,246.87c-4.38-.37-8.92.54-13.01,2.93l-76.21,43.94c-10.72,6.2-14.38,19.96-8.2,30.7,6.22,10.71,19.91,14.4,30.61,8.17l42.7-24.63v49.21c0,12.4,10.03,22.45,22.45,22.45s22.42-10.05,22.42-22.45v-87.98c0-11.81-9.16-21.51-20.75-22.34ZM138.91,205.67c.76-2.4,1.08-4.87,1.02-7.31-.09-1.76-.32-3.53-.82-5.29-1.51-5.48-5.08-10.38-10.44-13.46l-76.15-43.97c-10.75-6.19-24.45-2.51-30.63,8.22-6.23,10.72-2.55,24.41,8.2,30.61l42.55,24.61-42.55,24.54c-10.75,6.22-14.41,19.9-8.2,30.67,6.17,10.7,19.88,14.37,30.63,8.18l76.15-44c5.14-2.96,8.63-7.59,10.23-12.8Z"/>
        #     </svg>""", unsafe_allow_html=True)

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
    # ("question: ",question)
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

if not os.environ['REPLICATE_API_TOKEN']:
    st.warning("Please enter your Replicate API token")
    st.stop()
    
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

