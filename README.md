# langgraph-code-agent
A simple, rebust and elegant way to answer question with code generation and execution.

## Code Agent workflow Graph

![code-agent-workflow-graph](Langgraph_code_agent.png?raw=true "code-agent")


### Sample

##### User Question:
```
Get latest Microsoft Stock Price from Yahoo Finance using yfinance library
```
##### Coding Task:
--------------------------------------
```
The task is to retrieve the latest Microsoft (MSFT) stock price from Yahoo Finance using the yfinance library. The code defines a function `get_stock_price` that takes a stock ticker as input, fetches the historical data for the last day using `yfinance`, and returns the closing price. The `main` function calls `get_stock_price` with 'MSFT' as the ticker and prints the result. Error handling is included to catch potential exceptions during data retrieval.
```

##### Generated Code:
--------------------------------------
```
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

```
--------------------------------------
##### Final Answer:
```
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


### Inspiration
Hugginface smolagents
