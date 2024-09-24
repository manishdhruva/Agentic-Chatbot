# Chatbot for Vacation Requests: ASTUDIO ASSESSMENT

### Description

---

A prototype application that leverages LangChain's agent and tool modules to provide automated leave-related answers. Utilizing FAISS for efficient data retrieval and powered by GPT-3.5-turbo, this chatbot offers a streamlined user experience through a Streamlit-based interface.

Tools currently assigned:

1. Timekeeping Policies - A ChatGPT generated sample HR policy document. Embeddings were created for this doc using OpenAI’s *text-embedding-ada-002* model and stored in a FAISS index.
2. Employee Data - A csv file containing dummy employee data (e.g. name, supervisor, # of leaves etc). It's loaded as a pandas dataframe and manipulated by the LLM using LangChain's PythonAstREPLTool
3. Calculator - this is LangChain's calculator chain module, LLMMathChain
4. process_vacation_request(User Defined) - this is used to check leave eligibility based on blackout periods, probation, notice periods, and available vacation days. It will also update the employee's available vacation days upon approval.

### Instructions

---

#### Notes:

1. `employee_data.csv` - contains all the details of the new/existing employees.
2. `leave_policies.txt` - contains GPT-generated company policies for leaves.
3. `chatbot-notebook.ipynb` - contains code snippets for interactive notebook experience.
4. `/streamlit-APP` directory - contains a streamlit app for better UI experience.

#### Repo setup:

1. Install `Python 3.10` for your operating system.
2. Clone the repository.
3. Install dependencies using `pip install -r requirements.txt`.
4. Configure your API keys in  `streamlit-APP/chatbot_backend.py`.
   * By default, AzureOpenAI approach is used.
   * If you want to use OpenAI API keys directly, uncomment lines following 117 and 137 in the above mentioned `.py` file.
   * By default, user(employee) is set to "Manish Dhruva", you can replace it with other employees in the csv file at line 160.
   * The logic for unpaid-leave is not complete due to time constraints.
5. Run the application with `streamlit run streamlit-APP/chatbot_frontend.py`.

### Libraries Used

---

[LangChain](https://python.langchain.com/docs/tutorials/) - development frame work for building apps around LLMs.
[FAISS](https://ai.meta.com/tools/faiss/) - the vector database for storing the embeddings.
[Streamlit](https://streamlit.io/) - used for the front end. Lightweight framework for deploying python web apps.
