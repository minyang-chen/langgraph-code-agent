# /**
#  *
#  * Python Code Agent
#  * Created by minyang in 2025-03.06
#  * Copyright (c) 2026. All rights reserved.
#  *
#  */
#

from python_code_agent_v3 import app

if __name__ == '__main__':
    #question = "How do I merge two DataFrames based on a common column?"
    #question = "Get latest Microsoft Stock Price from Yahoo Finance using yfinance library."
    #question = "How do I fill missing values in a DataFrame with the mean of the column?"
    #question = "how to calculate the sum of all elements in a list using python?"
    #question = "list all the files in current directory"
    question = "Create an itinerary outlining all the essential details for a trip from Toronto to Niagara Falls."
    #question="I can come on Monday, but I forgot my passport so risk being delayed to Wednesday, is it possible to take me and my stuff to surf on Tuesday morning, with a cancellation insurance?"
    solution = app.invoke({"messages": [{"role": "user", "content": question}], "iterations": 0, "error": ""})

    # Print the final solution
    result = solution["generation"]
    print("======================================")
    print("<User Question>")
    print(question)
    print("--------------------------------------")
    print("<Coding Task>")
    print(result.task)
    #print(f"Code Imports:\n{result.imports}")
    print("--------------------------------------")
    print("<Generated Code>")
    print("```",result.code.strip(),"\n```")
    print("--------------------------------------")
    print(f"<Final Answer>")
    print("--------------------------------------")
    print(result.result)
    print("======================================")
    print(f"<Has error> {result.error}")
    