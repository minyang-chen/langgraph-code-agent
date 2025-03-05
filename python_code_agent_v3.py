# /**
#  *
#  * Python Code Agent
#  * Created by minyang in 2025-03.06
#  * Copyright (c) 2026. All rights reserved.
#  *
#  */
#
# Description: This is a code agent generate python code to answer user question.
#
import re
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, TypedDict
from langgraph.graph import END, StateGraph, START

## Data Model
class CodeSolution(BaseModel):
    """Schema for problem solution logic."""
    prompt: str = Field(description="Optimize code code generation user prompt to answer user question")
    reason: str = Field(description="Justification optimization of coding logic in python to solve user question")    

class PythonCode(BaseModel):
    """Schema for Python code solutions."""
    task: str = Field(description="Task Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    result: str = Field(description="Code execution result")
    error: str = Field(description="Code execution error")    

## Ollama OpenAI Model
## --------------------
client = OpenAI(base_url="http://192.168.0.30:11434/v1", api_key="ollama")
PARAM_MODEL_NAME="qwen2.5:14b"
PARAM_MODEL_TEMPERATURE=0
PARAM_MODEL_MAX_TOKENS=6000
PARAM_MODEL_TOP_P=1
PARAM_MODEL_FREQUENCY_PENALTY=0
PARAM_MODEL_PRESENCE_PENALTY=0     

# ## Google OpenAI Model
# ## --------------------
api_key = "" # Replace with your secret name
client = OpenAI(
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
    api_key=api_key,
)
PARAM_MODEL_NAME="gemini-2.0-flash-exp"

# the maximum number of iterations for the workflow. If the code fails the tests, the workflow will retry up to this limit.
PARAM_MAX_RETRY_ITERATIONS=3
PARAM_MAX_CODE_TIMEOUT=60

## Defining the Graph State
class GraphState(TypedDict):
    """
    Represents the state of the workflow graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : List of User Message and Assistant Message objects
        generation : Python code solution
        iterations : Number of tries
    """
    error: str
    messages: List[Dict[str, Any]]
    generation: PythonCode
    iterations: int

# ==========================
# Defining Nodes (Functions)
# ==========================

## Translation question to prompt Node
def optimize_prompt(state: GraphState) -> GraphState:
    """
    Optimize User Prompt for Python Code Generation.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with new generation.
    """
    print("## STEP 1: OPTIMIZE PROMPT")
    # State
    messages = state["messages"]
    iterations = state["iterations"]

    SYSTEM_PROMPT="""You are a coding assistant specialized in Python programming and problem solving tasks with coding. 
    Structure your answer in the following format:
    ---
    prompt: <Optimized code code generation user prompt to answer user question>
    reason: <Justification on optimization of coding logic in python to solve user question>
    ---
    """
    ## extract question
    if len(messages)==1:
        question = messages[0]["content"]
    ## creater user prompt
    USER_PROMPT=f"""Optimize following user prompt for generation of clean python coding logic to solve user question.\n
    If you need to use any external libraries, include a comment at the top of the code listing the required pip installations\n
    Provide justification why your response can solve the user question with step by step coding logic .\n\n
    
    USER QUESTION:\n {question}\n\n
    
    Respond only with one best user problem optimized prompt and reason for coding logic.\n
    Ensure generate code include main function to run the code.\n
    Ensure generate code exclude ask user provide input.\n    
    """
        
    print("### STEP 1.1: User Question:",question)                   
    print("### STEP 1.2: User Prompt:",USER_PROMPT)      
    user_messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]    
        
    try:
        completion = client.beta.chat.completions.parse(
            temperature=PARAM_MODEL_TEMPERATURE,
            max_tokens=PARAM_MODEL_MAX_TOKENS,
            top_p=PARAM_MODEL_TOP_P,
            model=PARAM_MODEL_NAME,
            messages=user_messages,
            response_format=CodeSolution,
        )
        code_response = completion.choices[0].message
        ## structure response
        if code_response.parsed:
            print("### STEP 1.3: LLM Optimized Prompt")                   
            print(code_response.parsed.prompt)            
            print("### STEP 1.4: LLM Justification")  
            print(code_response.parsed.reason)                                        
        elif code_response.refusal:
            print("### STEP 1.5: LLM Refusal")                               
            print(code_response.refusal)
    except Exception as e:
        print("### STEP 1.6: Error")                                       
        print(f"<LLM Error>: {e}")    
    
    ## code generation prompts
    CODEGEN_SYSTEM_PROMPT="""You are a coding assistant specialized in Python code generator. Respond only with complete executable Python code, no explanations or comments except for required pip installations at the top.\n
    If you need to use any external libraries, include a comment at the top of the code listing the required pip installations.\n
    Structure your answer in the following format:
    ---
    Task: <description of the solution>
    Imports: <required import statements>
    Code: <executable code block>
    ---
    """
    OPTIMIZED_USER_PROMPT=f"""
    {code_response.parsed.prompt}\n\n
    Reason: {code_response.parsed.reason}\n\n
    """
    user_messages=[
        {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
        {"role": "user", "content": OPTIMIZED_USER_PROMPT}
    ]   
    print("### STEP 1.7: Save Optimized Prompt")                                            
    # Increment
    iterations = iterations + 1   
    return {"generation": "", "messages": user_messages, "iterations": 0, "error": ""}            

## Code Generation Node
def code_generate(state: GraphState) -> GraphState:
    """
    Generate a Python code solution.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with new generation.
    """
    print("## STEP 2: GENERATING CODE")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    ## We have been routed back to generation with an error
    if error == "yes":
        print(f"### STEP 2.1: Retry {iterations}")                
        messages.append({"role": "user", "content": "Now, try again. Generate a complete python execution code with a description, imports, and code block:"})        
    else:
        print(f"### STEP 2.2: process iteration {iterations}")        

    try:
        ## Solution
        completion = client.beta.chat.completions.parse(
            temperature=PARAM_MODEL_TEMPERATURE,
            max_tokens=PARAM_MODEL_MAX_TOKENS,
            top_p=PARAM_MODEL_TOP_P,
            # frequency_penalty=PARAM_MODEL_FREQUENCY_PENALTY,
            # presence_penalty=PARAM_MODEL_PRESENCE_PENALTY,            
            model=PARAM_MODEL_NAME,
            messages=messages,
            response_format=PythonCode,
        )
        code_response = completion.choices[0].message
        ## structure response
        if code_response.parsed:
            print("### STEP 2.3: Code Generation Response:")            
            print("<LLM Task>:",code_response.parsed.task)
            print("<Code Imports>:",code_response.parsed.imports)        
            print("<Code Block>:",code_response.parsed.code)               
            code_solution = PythonCode(task=code_response.parsed.task, imports=code_response.parsed.imports, code=code_response.parsed.code, result="", error="no")
        elif code_response.refusal:
            print("### STEP 2.4: LLM Refusal")
            print(code_response.refusal)            
            code_solution = PythonCode(task=code_response.parsed.description, imports="",code="", result="refusal", error="Yes")            
    except Exception as e:
        print("### STEP 2.4: LLM Error")        
        print(f"<LLM Error>: {e}")    
        code_solution = PythonCode(task=str(e.args), imports="",code="", result="", error="Yes")                    
    
    # save response
    messages.append({"role": "assistant", "content": f"Task: {code_solution.task} \n Imports: {code_solution.imports} \n Code: {code_solution.code}"})        
    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": error}

## Import Check Node
def import_check(state: GraphState):
    """
    Install necessary libraries by detecting imports and pip comments.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with error status.
    """
    print("## STEP 3: CHECK IMPORTS ***")    
    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code    
    
    #print("BEFORE:\n", code)
    # import libraries    
    import subprocess
    import pkg_resources
    import sys
    ## -----------------------
    ## Remove SPECIAL COMMENTS
    ## -----------------------
    ## Remove GOOGLE_MAPS_API_KEY text
    #text_to_remove = "Replace 'YOUR_GOOGLE_MAPS_API_KEY' with your actual Google Maps API key."
    #code = re.sub(re.escape(text_to_remove), '', code)
    # Extract libraries from import statements
    imports = re.findall(
        r'^\s*(?:import|from) ([\w\d_]+)', code, re.MULTILINE)

    # handlecase of empty imports in code
    if len(imports)==0:
        print("### STEP 3.1: No imports found.")
        imports = re.findall(
            r'^\s*(?:import|from) ([\w\d_]+)', code_solution.imports, re.MULTILINE)

    # exclude default python libraries
    default_libraries = ['os', 'sys', 're', 'math', 'random', 'datetime', 'time', 'json', 'csv', 'requests', 'urllib', 'sqlite3']
    imports = [lib for lib in imports if lib not in default_libraries]
        
    # Extract libraries from pip install comments
    libraries = re.findall(r'#\s*pip install\s+([\w-]+)', code)

    # Combine both sources
    all_libraries = set(imports + libraries)
    if all_libraries:
        print("### STEP 3.2: Checking required libraries...")
        print(all_libraries)        
        for lib in all_libraries:
            package_name = lib.replace('-', '_')
            try:
                pkg_resources.require(package_name)
                print(f"{lib} is already installed.")
            except pkg_resources.DistributionNotFound:
                try:
                    print(f"### STEP 3.2.1: Installing {lib}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                except Exception as e:
                    print(f"### STEP 3.2.2: Error installing {lib}: {e}")
    else:
        print("### STEP 3.2: No missing imports found.")              
    print("### STEP 3.3: All required libraries are installed.")

    # update clean up code
    code_solution.code = code
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }    

## Validate Generated Code Node
def code_check(state: GraphState) -> GraphState:
    """
    Check the generated Python code.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with error status.
    """
    print("## STEP 4: CHECK GENERATED CODE")        
    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # handle code block start with ```python`
    if code.find('python')>0:        
        # Extract code block
        imports = re.sub(r'^```python\n|^```\n|```$', '',code_solution.imports, flags=re.MULTILINE)
        code = re.sub(r'^```python\n|^```\n|```$', '',code_solution.code, flags=re.MULTILINE)
        code_lines = code.split('\n')
        # remove duplicate import lines at the beginning of the code block
        while code_lines and not (code_lines[0].startswith('import') or code_lines[0].startswith('from') or code_lines[0].startswith('#') or code_lines[0].startswith('---')):
            code_lines.pop(0)
        code ='\n'.join(code_lines)

    # Execution Code
    if code.startswith('import'):
        execution_code =  code
    else:
        execution_code = imports + "\n\n" + code           

    # code block
    print("```<code>")    
    print(execution_code)    
    print("</code>```")      
    
    # update tested code
    code_solution.code = execution_code
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }

## Improvde Code include Environment Settings Node 
def setup_code(state: GraphState) -> GraphState:
    """
    Setup the generated code with environment parameters.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with new generation.
    """
    print("## STEP 5: CODE SETUP")

    # State
    code_solution = state["generation"]    
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]
        
    if "API_KEY" not in code_solution.code:
        print("### STEP 5.1: Not API Key required:")                        
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": error}            
    else:
        USER_PROMPT=f"""I have a list of supported API Keys store as Environment variables, they are:\n
        GOOGLE_MAPS_API_KEY, TAVILY_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY,LOGFIRE_TOKEN\n\n
        
        when generated code use any of above API Keys then updated code to dynamically replace the API KEY with environment variables otherwise leave the code as-is:\n
        
        {code_solution.code} \n
        
        Ensure optimize code include main function to run the code, all imports libraries and try and catch error handlers.\n
        Respond must satify following justification:\n
        {code_solution.task}
        """
        print("### STEP 5.2: User Prompt:")                
        print(USER_PROMPT,"\n")
        
        messages.append(
            {"role": "assistant", "content": USER_PROMPT}
        ) 
        try:
            ## Solution
            completion = client.beta.chat.completions.parse(
            temperature=PARAM_MODEL_TEMPERATURE,
            max_tokens=PARAM_MODEL_MAX_TOKENS,
            top_p=PARAM_MODEL_TOP_P,
            # frequency_penalty=PARAM_MODEL_FREQUENCY_PENALTY,
            # presence_penalty=PARAM_MODEL_PRESENCE_PENALTY,            
            model=PARAM_MODEL_NAME,
            messages=messages,
            response_format=PythonCode)
            ## structure response        
            code_response = completion.choices[0].message
            if code_response.parsed:
                print("### STEP 5.2.1: Code Generation Response:")            
                print("<LLM Task>:",code_response.parsed.task)
                print("<Code Imports>:",code_response.parsed.imports)        
                print("<Code Block>:",code_response.parsed.code)               
                code_solution = PythonCode(task=code_response.parsed.task, imports=code_response.parsed.imports, code=code_response.parsed.code, result="", error="no")
            elif code_response.refusal:
                print("### STEP 5.2.2: LLM Refusal")
                print(code_response.refusal)            
                code_solution = PythonCode(task=code_response.parsed.description, imports="",code="", result="refusal", error="Yes")            
        except Exception as e:
            print("### STEP 5.2.3: LLM Error")        
            print(f"<LLM Error>: {e}")    
            code_solution = PythonCode(task=str(e.args), imports="",code="", result="", error="Yes")                    
        
        messages.append({"role": "assistant", "content": f"Task: {code_solution.task} \n Imports: {code_solution.imports} \n Code: {code_solution.code}"})        
        # Increment
        #iterations = iterations + 1
        return {"generation": code_solution, "messages": messages, "iterations": iterations, "error": error}    

## Run Code and Collect Result Node 
def code_execution(state: GraphState) -> GraphState:
    """
    Execute generated Python code.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated state with error status.
    """
    print("## STEP 6: EXECUTE CODE ")             
    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]
    
    try:
        import sys
        import os
        import subprocess
        import tempfile        
        
        # Get Execution Code    
        execution_code=code_solution.code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(execution_code)
            temp_file_path = temp_file.name
            print("### STEP 6.1: Generated Code file path:", temp_file_path)
        try:
            result = subprocess.run(
                [sys.executable, temp_file_path], capture_output=True, text=True, timeout=PARAM_MAX_CODE_TIMEOUT)
            output = result.stdout
            error = result.stderr
            if result.returncode == 0:
                # No errors                
                print("### STEP 6.3: Execution Status: SUCCESS! ***")            
                print("### STEP 6.2: Code Output:")
                print("======================================")            
                print(output)
            else:            
                print("### STEP 6.2: Execution Error:", result.returncode)
                print(output, error)                
                raise Exception(error)             
        except subprocess.TimeoutExpired:
            output = ""
            error = f"Execution timed out after {PARAM_MAX_CODE_TIMEOUT} seconds."
            print("### STEP 6.2: Execution Error:",error)            
            raise Exception(error)
        finally:
            os.unlink(temp_file_path)
    except Exception as e:
        print("### STEP 6.3: Execution Error: FAILED! ***")
        print("Exception:",e)
        messages.append({"role": "user", "content": f"Your solution failed the import test: {e}"})        
        # record code result
        code_solution.result = e.args
        code_solution.error = "yes"
        print("### STEP 6.3: Error:")            
        print(e)                                
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }
    # response message    
    messages.append({"role": "assistant", "content": f"code execution result:\n {output}"})    
    
    # record code result
    code_solution.result = output
    code_solution.error = "no"
    
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }

## Defining Edges
def decide_to_finish(state: GraphState) -> str:
    """
    Determines whether to finish.

    Args:
        state (GraphState): The current graph state.

    Returns:
        str: Next node to call.
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == PARAM_MAX_RETRY_ITERATIONS:
        print("## STEP 7: FINISH ")
        return "end"
    else:
        print("## STEP 7: RE-TRY")
        return "code_generate"

# ==========================
# Building Code Agent Graph
# ==========================
# We build the workflow graph by adding nodes and edges. The graph starts with the generate node, moves to the check_code node, and then decides whether to finish or retry based on the code validation results.

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("optimize_prompt", optimize_prompt)  # Optimize user prompt and solution logic 
workflow.add_node("code_generate", code_generate)  # Generate Code
workflow.add_node("check_import", import_check)  # Check imports
workflow.add_node("check_code", code_check)  # Check code
workflow.add_node("setup_code", setup_code)  # Setup code with environment parameters
workflow.add_node("run_code", code_execution)  # Run generated code

# Build graph
workflow.add_edge(START, "optimize_prompt")
workflow.add_edge("optimize_prompt", "code_generate")
workflow.add_edge("code_generate", "check_import")
workflow.add_edge("check_import", "check_code")
workflow.add_edge("check_code", "setup_code")
workflow.add_edge("setup_code", "run_code")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "code_generate": "code_generate",
    },
)
workflow.add_conditional_edges(
    "run_code",
    decide_to_finish,
    {
        "end": END,
        "code_generate": "code_generate",
    },
)

# Compile the workflow
app = workflow.compile()

print("\nCode Agent Graph:\n")
ascii_graph=app.get_graph(xray=1).draw_ascii()
print("```",ascii_graph,"```\n")

## required library to generate png
# sudo apt-get install graphviz
# pip install graphviz
## 
# png_graph = app.get_graph(xray=1).draw_png(output_file_path="./codeagent_graph.png")
# print("```",png_graph,"```\n")

# from IPython.display import Image, display
# display(Image(app.get_graph(xray=1).draw_mermaid_png()))

#########################
## Testing the Workflow
# =======================

# ## Example 3: Filling Missing Values
# ## ------------------------------------------
# question = "How do I fill missing values in a DataFrame with the mean of the column?"
# #prompt = f"Generate Python code to {question}\n. If you need to use any external libraries, include a comment at the top of the code listing the required pip installations."
# solution = app.invoke({"messages": [HumanMessage(content=question)], "iterations": 0, "error": ""})

# #question = "How do I merge two DataFrames based on a common column?"
# #question = "Get latest Microsoft Stock Price from Yahoo Finance using yfinance library."
# #question = "How do I fill missing values in a DataFrame with the mean of the column?"
# #question = "how to calculate the sum of all elements in a list using python?"
# #question = "list all the files in current directory"
# question = "Create an itinerary outlining all the essential details for a trip from city Orangeville to Niagara Falls."
# solution = app.invoke({"messages": [{"role": "user", "content": question}], "iterations": 0, "error": ""})

# # Print the final solution
# result = solution["generation"]
# print("======================================")
# print("<User Question>")
# print(question)
# print("--------------------------------------")
# print("<Coding Task>")
# print(result.task)
# #print(f"Code Imports:\n{result.imports}")
# print("--------------------------------------")
# print("<Final Code>")
# print("```",result.code.strip(),"\n```")
# print("--------------------------------------")
# print(f"<Final Result>")
# print("--------------------------------------")
# print(result.result)
# print("======================================")
# print(f"<Has error> {result.error}")

# We test the workflow by asking it to generate solutions for three different Pandas-related questions. The final solution is printed, including the description, imports, and code block.

## Example 1: Grouping and Calculating Mean
## ------------------------------------------
# This example demonstrates how to group a DataFrame by a specific column and calculate the mean of another column. Grouping and aggregation are common operations in data analysis, and Pandas provides a straightforward way to perform these tasks using the groupby method.

# question = "How do I group a DataFrame by a column and calculate the mean of another column?"
# solution = app.invoke({"messages": [HumanMessage(content=question)], "iterations": 0, "error": ""})

# # # Print the final solution
# # print("======================================")
# # result = solution["generation"]
# # print(f"Task:\n{result.description}\n")
# # print(f"Imports:\n{result.imports}\n")
# # print(f"Code:\n{result.code}\n")
# # print(f"Result:\n{result.result}\n")
# # print(f"Error:\n{result.error}\n")
# print("======================================")
# # Print response messages
# iterations = solution["iterations"]
# print("Iterations:", iterations)
# for message in solution["messages"]:
#     if isinstance(message,HumanMessage):
#         print("<User>")
#         print(message.content)        
#         print("</User>")        
#     if isinstance(message,AIMessage):
#         print("<AI>")        
#         print(message.content)        
#         print("</AI>")                        

# ## Example 2: Merging DataFrames
# ## ------------------------------------------
# # This example shows how to merge two DataFrames based on a common column. Merging is a fundamental operation when working with multiple datasets, and Pandas' merge function allows for flexible joining of DataFrames using different types of joins (e.g., inner, outer, left, right).

# question = "How do I merge two DataFrames based on a common column?"
# solution = app.invoke({"messages": [HumanMessage(content=question)], "iterations": 0, "error": ""})

# # Print the final solution
# result = solution["generation"]
# print(f"Task:\n{result.description}\n")
# print(f"Imports:\n{result.imports}\n")
# print(f"Code:\n{result.code}\n")
# print(f"Result:\n{result.result}\n")
# print(f"Error:\n{result.error}\n")

