import os
from config import OPENAI_API_KEY
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langsmith import utils
import logging
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence

utils.tracing_is_enabled()

# Creating the prompt to instruct the model on how/what to do to generate the SQL queries
prompt = ChatPromptTemplate.from_template("""You are a expert at generating SQL queries for the following table schema.

Table : products
Columns : id(auto increment, primary key, integer), created_at(timestamp), name(varchar), description(varchar), price(numeric), stock(integer)

And the following question: {question}

Some sample user queries and their corresponding SQL queries for your understanding:
1. What are the names of all the products?
SQL query: SELECT name FROM products

2. What are the names of all the products that cost more than $100?
SQL query: SELECT name FROM products WHERE price > 100

3. What are the names of all the products that cost more than $100 and have a stock quantity of 10 or more?
SQL query: SELECT name FROM products WHERE price > 100 AND stock >= 10

Please keep in mind that, certain functions and methods can only be used on some columns, based on their datatype. 
So, knowing the datatype of the columns is really important.

If you dont know the answer, just say you dont know.

Generate a SQL query to answer this question.
""")

# Defining the LLM that will be used to answer the user query
model = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model="gpt-3.5-turbo")

# Defining the output parser to parse the output of the LLM and exrtact the content out of the AIMessage object response
def outputParser(response):
    logging.info("NL2SQL RAG output: Parsing the response...")
    return response.content


# Chain
nl2sql_chain = ({"question": RunnablePassthrough()} | prompt | model | outputParser)

# Invoking the chain with a sample question
if __name__ == "__main__":
    response = nl2sql_chain.invoke("What are the names of all the products whose name contains 'ea' and have a stock quantity of 10 or more?")
    print(response)