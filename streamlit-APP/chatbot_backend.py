# load tokenizers for prepping policies
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load core modules
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, FAISS
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chains import RetrievalQA

# load agents and tools modules
import pandas as pd
from azure.storage.filedatalake import DataLakeServiceClient
from io import StringIO
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain
import re

# Function to process vacation requests using LLM to extract dates
def process_vacation_request(user_input):
    # Step 1: Extract vacation dates using regex (yyyy-mm-dd format)
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    vacation_dates = re.findall(date_pattern, user_input)
    
    # Check if both start and end dates are found
    if len(vacation_dates) != 2:
        return "Invalid date format. Please provide dates in 'YYYY-MM-DD to YYYY-MM-DD' format."

    start_date_str, end_date_str = vacation_dates
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        return "Error parsing dates. Ensure the format is 'YYYY-MM-DD to YYYY-MM-DD'."

    # Step 2: Retrieve employee details from df using the Python REPL tool
    employee_query = f"df[df['name'] == '{user}']"
    employee_data = python.run(employee_query)
    
    # Check if the employee exists in the dataframe
    if 'None' in employee_data or 'Empty DataFrame' in employee_data:
        return f"Employee {user} not found."

    # Step 3: Parse employee details
    available_vacation_days_str = agent.run(f"Extract 'available_vacation_days' from this data: {employee_data}")

    # Use regex to extract only the numeric value
    available_vacation_days = int(re.search(r'\d+', available_vacation_days_str).group())

    probation_completed = agent.run(f"Extract 'probation_completed' from this data: {employee_data}").lower()
    under_notice_period = agent.run(f"Extract 'under_notice_period' from this data: {employee_data}").lower()

    # Step 4: Assess vacation eligibility based on company policies
    blackout_dates = pd.to_datetime(["2024-10-06", "2024-10-12"])  # Example blackout dates
    requested_range = pd.date_range(start=start_date, end=end_date)

    if any(date in blackout_dates for date in requested_range):
        return "Requested dates fall within company blackout periods. Please choose different dates."
    
    if probation_completed != 'true':
        return "You are still in the probation period and are not eligible for vacation requests."
    
    if under_notice_period == 'true':
        return "You cannot request a vacation while under a notice period."

    # Step 5: Check if the requested vacation days exceed available vacation days
    requested_days = (end_date - start_date).days + 1
    if requested_days > available_vacation_days:
        return f"You only have {available_vacation_days} vacation days left, but you requested {requested_days} days. Consider taking unpaid leave."

    # Step 6: Update employee's vacation days in the dataframe
    new_vacation_days = available_vacation_days - requested_days
    update_query = f"df.loc[df['name'] == '{user}', 'available_vacation_days'] = {new_vacation_days}"
    python.run(update_query)

    # Save changes to CSV
    python.run("df.to_csv('../employee_data.csv', index=False)")  

    # Step 7: Return confirmation message
    approval_message = f"Your vacation request from {start_date_str} to {end_date_str} has been approved. Your available vacation days have been updated to {new_vacation_days}."
    return approval_message

# Tokenizer and text splitter for policies
tokenizer = tiktoken.get_encoding('p50k_base')
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# Load the policy file
doc_path = r"leave_policies.txt"
with open(doc_path, 'r') as f:
    contents = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

texts = text_splitter.split_text(contents)
metadatas = [{"text": text} for text in texts]

# Initialize Azure-OpenAI embeddings
embed = OpenAIEmbeddings(
                deployment="<your deployment name>",
                model="text-embedding-ada-002",
                openai_api_key='<your azure openai api key>',
                openai_api_base="<your api base>",
                openai_api_type="azure",
                openai_api_version="2023-03-15"
            )

# Uncomment below snippet for using OPENAI API KEY directly
# embed = OpenAIEmbeddings(
#                 model = 'text-embedding-ada-002',
#                 openai_api_key="<your openai api key from from platform.openai.com>",
#             )

# Create FAISS vectorstore
vectorstore = FAISS.from_texts(texts, embed, metadatas=metadatas)

# Initialize Azure LLM
llm = AzureChatOpenAI(    
    deployment_name="<your deployment name>", 
    model_name="gpt-35-turbo", 
    openai_api_key='<your openai api key>',
    openai_api_version = '2023-03-15-preview', 
    openai_api_base='<your api base>',
    openai_api_type='azure',
    temperature=0.0
    )

# Uncomment below snippet for using OPENAI API KEY directly
# llm = ChatOpenAI(    
#     openai_api_key="<your openai api key from from platform.openai.com>", 
#     model_name="gpt-3.5-turbo", 
#     temperature=0.0
#     )


# Initialize vectorstore retriever object for policy
timekeeping_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# Load employee data into a dataframe
df = pd.read_csv("../employee_data.csv") # load employee_data.csv as dataframe
python = PythonAstREPLTool(locals={"df": df}) # set access of python_repl tool to the dataframe

# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# create variables for f strings embedded in the prompts
user = 'Manish Dhruva' # set user
df_columns = df.columns.to_list() # print column names of df

# prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
tools = [
    Tool(
        name = "Timekeeping Policies",
        func=timekeeping_policy.run,
        description="""
        Useful for when you need to answer questions about employee timekeeping policies.

        <user>: What is the policy on unused vacation leave?
        <assistant>: I need to check the timekeeping policies to answer this question.
        <assistant>: Action: Timekeeping Policies
        <assistant>: Action Input: Vacation Leave Policy - Unused Leave
        ...
        """
    ),
    Tool(
        name = "Employee Data",
        func=python.run,
        description = f"""
        Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        <user>: How many Sick Leave do I have left?
        <assistant>: df[df['name'] == '{user}']['sick_leave']
        <assistant>: You have n sick leaves left.              
        """
    ),
    Tool(
        name = "Calculator",
        func=calculator.run,
        description = f"""
        Useful when you need to do math operations or arithmetic.
        """
    ),
    Tool(
        name="Vacation Request Processor",
        func=process_vacation_request,
        description = f"""
        Useful for processing vacation requests. It will automatically extract the vacation dates from the input,
        retrieve the employee data from the dataframe, and check eligibility based on blackout periods, probation,
        notice periods, and available vacation days. The tool will update the employee's available vacation days upon approval.

        Example usage:
        <user>: I'd like to take vacation from 2024-12-20 to 2024-12-26.
        <assistant>: I'll check your eligibility and process the request.
        """
    )
]

# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
agent_kwargs = {'prefix': f'You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:'}


# initialize the LLM agent
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         agent_kwargs=agent_kwargs
                         )

# define q and a function for frontend
def get_response(user_input):
    response = agent.run(user_input)
    return response