# langgraph-code-agent
A simple, rebust and elegant way to answer question with code generation and execution.

## Code Agent workflow Graph

![code-agent-workflow-graph](Langgraph_code_agent.png?raw=true "code-agent")


### When to use Code Agent?

***Choose Simplicity and Robustness***
Ideal for users seeking simplicity in setup and use while still requiring robust solutions. 

***Continue Adapt Changing Requirement***
When pre-determined workflows fall short due to evolving business needs, code agents offer the flexibility to adapt without constant reprogramming 

***Improved Efficiency in Complex Workflows***
Traditional multi-step agents can become overly complex with numerous conditional statements (if/else). Code agents mitigate this by dynamically generating code, thus handling complexity more cleanly and efficiently.

***Simplify Task Automation with Multiple Tools***
For tasks that require interaction with multiple tools or services, code agents excel by automating the process of tool configuration and execution through generated code, reducing manual intervention.

***Make Intelligent Decision-Making***
By leveraging large language models (LLMs), code agents incorporate an intelligent decision-making layer, allowing them to choose which actions to execute and how, beyond simple rule-based automation

#### Sample-1
Question:  ***how many r in strawberry?***

```
Coding Task:
Count the number of times the letter 'r' appears in the word 'strawberry'.
--------------------------------------
Generated Code:
--------------------------------------
def main():
    word = "strawberry"
    count = 0
    # Iterate through the word and count the occurrences of 'r'
    for char in word:
        if char == 'r':
            count += 1
    print(f"The letter 'r' appears {count} times in the word 'strawberry'.")

if __name__ == "__main__":
    main() 
--------------------------------------
<Final Answer>
--------------------------------------
The letter 'r' appears 3 times in the word 'strawberry'.
======================================
<Has error> no
```

### Sample-2


User Question: ***Get latest Microsoft Stock Price from Yahoo Finance using yfinance library***

```
##### Coding Task:
--------------------------------------

The task is to retrieve the latest Microsoft (MSFT) stock price from Yahoo Finance using the yfinance library. The code defines a function `get_stock_price` that takes a stock ticker as input, fetches the historical data for the last day using `yfinance`, and returns the closing price. The `main` function calls `get_stock_price` with 'MSFT' as the ticker and prints the result. Error handling is included to catch potential exceptions during data retrieval.

##### Generated Code:
--------------------------------------

import yfinance as yf

def get_stock_price(ticker):
	"""Retrieves the latest stock price from Yahoo Finance.

	Args:
    	ticker (str): The stock ticker symbol (e.g., 'MSFT').

	Returns:
    	float: The latest stock price, or None if an error occurs.
	"""
	try:
    	import yfinance as yf
    	msft = yf.Ticker(ticker)
    	data = msft.history(period="1d")
    	if not data.empty:
        	return data['Close'][-1]
    	else:
        	return None
	except Exception as e:
    	print(f"An error occurred: {e}")
    	return None

def main():
	"""Main function to execute the stock price retrieval.
	"""
	ticker_symbol = "MSFT"
	latest_price = get_stock_price(ticker_symbol)

	if latest_price:
    	print(f"The latest stock price for {ticker_symbol} is: {latest_price:.2f}")
	else:
    	print(f"Could not retrieve the stock price for {ticker_symbol}.")
if __name__ == "__main__":
	main()

--------------------------------------
##### Final Answer:

The latest stock price for MSFT is: ***388.61***
```

### Prepare Environment
```
# support generation of graph workflow PNG on linux 
sudo apt-get install graphviz

# libraries 
pip install openai
pip install graphviz
pip install -qU langgraph
pip installl pydantic

# optional
pip install -qU langchain-openai
pip install -qU langchain_community
```

### Code Agent Usage
```
from python_code_agent_v3 import app

if __name__ == '__main__':
    question = "list all the files in current directory"
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
```

### Contribution
Open for contribution

### When to avoid Code Agent?

***Fixed Predetermined Workflow***
When your workflow is already set and does not require flexibility or changes, a Code Agent may not be necessary.
Processes Requiring Human In the loop
If your workflow involves steps where human decision-making or approvals are essential, relying on a Code Agent might not be suitable.

***Tight Control on Tool or API Calling Usage***
In scenarios where there are limitations on how tools or APIs can be utilized within your system—such as restricted access or regulatory constraints—a Code Agent may face challenges in automating these calls effectively.

***Static Workflow and Decision Making***
If the logic and decision-making processes in your workflow remain unchanged over time, employing a dynamic agent like a Code Agent becomes unnecessary.


### Inspiration
Hugginface smolagents
