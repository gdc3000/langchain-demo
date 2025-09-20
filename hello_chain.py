import pandas as pd
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# Function to extract the code block from the response
def extract_code_block(text: str) -> str | None:
    """
    Pull the LAST ```python ... ``` block from LLM output.
    Falls back to any ``` ... ``` if no language is given.
    """
    # Find all python code blocks first
    python_matches = re.findall(r"```python\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if python_matches:
        return python_matches[-1].strip()  # Return the last one
    
    # Fall back to any code blocks
    all_matches = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if all_matches:
        return all_matches[-1].strip()  # Return the last one
    
    return None

# Execute the code safely
def execute_code(code: str):
    _BLOCKED_SUBSTRINGS = [
        "import os", "import sys", "import subprocess", "import shlex",
        "import pathlib", "from os", "from sys", "from subprocess",
        "open(", "exec(", "eval(", "__import__", "shutil", "requests",
    ]

    for substring in _BLOCKED_SUBSTRINGS:
        if substring in code:
            raise ValueError(f"Blocked substring found: {substring}")

    """Execute the code safely"""
    exec(code)


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("data/OnlineRetail.csv", engine='python', encoding='utf-8', encoding_errors="ignore")

    # Load the model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build the prompt
    prompt = ChatPromptTemplate.from_template(
        """Assume that a dataset with info provided below exists. 
           Return only a fenced Python block to perform the analyis requested by the user.
           Your code does not need to load the dataset into the dataframe.
           Your code should assume the full dataset exists and has already been loaded into the variable df as a pandas dataframe prior to your code being executed.
           The sample dataset is for reference only.
           Your python code should be executable and should not raise any errors.
           Your python code should print all results in a human-readable format with text that explains the results.
        
        Dataset Info:
        - Shape: {df_shape}
        - Columns: {df_columns}
        - Sample data:
        {df_sample}
        
        User Request: {text}
        
        Please provide Python code using pandas to fulfill this request."""
    )

    # Build the chain
    chain = prompt | llm
    
     # Get user request
    user_request = input("Enter your analysis request: ").strip()
    if not user_request:
        user_request = "Create a simple analysis showing the top 5 countries by total sales"

    # Build the info
    df_info = {
        "df_shape": str(df.shape),
        "df_columns": list(df.columns),
        "df_sample": df.head().to_string(),
        "text": user_request
    }

    # Run the chain
    print("🔍 Analyzing request:", user_request)
    response = chain.invoke(df_info)
    content = response.content

    print("\n📊 Full response:")
    print("=" * 50)
    print(content)

    print("\n📊 Generated Code extracted from the result:")
    print("=" * 50)
    code = extract_code_block(content)
    print(code)

    print("\n📊 Run code:")
    print("=" * 50)
    execute_code(code)


    # Enable follow-up requests
    # Build session history
    session_history = []
    session_history.append(("User", user_request))
    session_history.append(("Assistant", response.content))
    while True:   
        # Get follow-up request
        followup = input("\nAsk a follow-up (or just press Enter to quit): ").strip()
        if not followup:
            break
        
        session_history.append(("User", followup))
        history_text = "\n".join(
            f"{role}: {msg}" for role, msg in session_history
        )

        # Build follow-up prompt
        followup_prompt = ChatPromptTemplate.from_template(
            """Here is the full conversation so far:

            {history_text}

            Please answer the follow-up using the dataset context. 
            If asked to perform new analysis, instruct the user to create a new request."""
        )

        followup_chain = followup_prompt | llm
        follow_response = followup_chain.invoke({
            "history_text": history_text
        })

        # Check if follow-up response contains code block
        followup_code = extract_code_block(follow_response.content)
        if followup_code:
            print("\n📊 Executing code from follow-up:")
            print("=" * 50)
            try:
                execute_code(followup_code)
            except Exception as e:
                print(f"❌ Error executing follow-up code: {e}")
        else:
            print("\n🔁 Follow-up response:")
            print("=" * 50)
            print(follow_response.content)

        session_history.append(("Assistant", follow_response.content))
