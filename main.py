from google import genai
from google.genai import types
import os
import re
import json
from botasaurus.browser import Driver
from typing import List, Dict, Any
import time
from dotenv import load_dotenv
import tempfile

load_dotenv()

# Initialize the Google GenAI client with API key
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

driver = Driver()

def filterhtml(html: str) -> str:
    # Remove all script and style tags
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    # Remove comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    # Compress whitespace
    html = re.sub(r'\s+', ' ', html)
    return html

def get_page_state():
    page_html = filterhtml(driver.page_html)
    screenshot = next(tempfile._get_candidate_names()) + '.png'
    driver.save_screenshot(screenshot)
    screenshot = "./output/screenshots/" + screenshot
    
    # Upload screenshot to Gemini API
    uploaded_file = client.files.upload(file=screenshot)
    
    # Remove local screenshot file
    os.remove(screenshot)
    
    return page_html, uploaded_file

# Define browser action tools for Gemini
def get_browser_tools():
    """Define the browser automation tools for Gemini function calling."""
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="click",
                    description="Click on an element using a CSS selector",
                    parameters={
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "CSS selector to identify the element to click"
                            }
                        },
                        "required": ["selector"]
                    }
                ),
                types.FunctionDeclaration(
                    name="type",
                    description="Type text into the current focused element or page",
                    parameters={
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "CSS selector to identify the element to type into"
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to type"
                            }

                        },
                        "required": ["selector", "text"]
                    }
                ),
                types.FunctionDeclaration(
                    name="get_text",
                    description="Get the text content of an element using a CSS selector",
                    parameters={
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "CSS selector to identify the element"
                            }
                        },
                        "required": ["selector"]
                    }
                ),
                types.FunctionDeclaration(
                    name="stop_browsing",
                    description="Stop browsing and return the current result. Use this when the goal has been achieved or when you need to stop the browsing session.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "string",
                                "description": "A summary of what was found or accomplished"
                            }
                        },
                        "required": ["result"]
                    }
                )
            ]
        )
    ]

def execute_tool_call(function_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a browser action tool call."""
    if function_name == "click":
        return driver.click(arguments["selector"])
    elif function_name == "type":
        return driver.type(arguments["selector"], arguments["text"])
    elif function_name == "get_text":
        return driver.get_text(arguments["selector"])
    elif function_name == "stop_browsing":
        return arguments.get("result", "Browsing stopped")
    else:
        raise ValueError(f"Unknown function: {function_name}")

def prompt_and_parse_tool_calls(prompt: str, page_html: str = None, screenshot_file = None, conversation_history = None) -> tuple:
    """
    Prompt Gemini with browser tools and parse tool calls from the response.
    
    Args:
        prompt: The user prompt/instruction
        page_html: Optional HTML content of the current page
        screenshot_file: Optional uploaded screenshot file from Gemini
        conversation_history: Optional list of previous messages in the conversation
    
    Returns:
        Tuple of (response, tool_calls) where tool_calls is a list of parsed tool calls
    """
    # Build the content parts for the prompt
    # contents should be a list of parts (strings, Part objects, or file references)
    contents = [prompt]
    
    # Add page context if available
    if page_html:
        contents.append(f"\nCurrent page HTML:\n{page_html}")
    
    # Add screenshot if available (file reference from upload)
    if screenshot_file:
        contents.append(screenshot_file)
    
    # Add conversation history if provided
    if conversation_history:
        # conversation_history should be a list of Content objects or parts
        contents = conversation_history + contents
    
    # Generate response with tools
    tools = get_browser_tools()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY)
            )
        )
    )
    
    # Parse tool calls from response
    tool_calls = []
    
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                # Parse arguments
                args = {}
                if hasattr(function_call, 'args'):
                    if isinstance(function_call.args, dict):
                        args = function_call.args
                    elif hasattr(function_call.args, '__dict__'):
                        args = dict(function_call.args)
                    else:
                        # Try to parse as JSON string if needed
                        try:
                            if isinstance(function_call.args, str):
                                args = json.loads(function_call.args)
                        except:
                            pass
                
                tool_calls.append({
                    "name": function_call.name,
                    "arguments": args
                })
    
    return response, tool_calls

def execute_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Any]:
    """
    Execute a list of tool calls and return their results.
    
    Args:
        tool_calls: List of tool calls with 'name' and 'arguments' keys
    
    Returns:
        List of results from executing each tool call
    """
    results = []
    for tool_call in tool_calls:
        try:
            result = execute_tool_call(tool_call["name"], tool_call["arguments"])
            results.append({
                "function_name": tool_call["name"],
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "function_name": tool_call["name"],
                "success": False,
                "error": str(e)
            })
    return results

def main():
    """
    Main function that combines all browser automation components.
    Uses Gemini AI to analyze web pages and execute browser actions to achieve a goal.
    Runs in a loop with a maximum number of steps.
    """
    GOAL = "Navigate to hacker news and find the most upvoted story"
    MAX_STEPS = 6
    
    prompt_template = f"""You are an AI browser automation assistant. Your goal is to: {GOAL}

You have access to browser automation tools:
1. click(selector) - Click on elements using CSS selectors
2. type(text, selector) - Type text into input fields
3. get_text(selector) - Extract text from elements
4. stop_browsing(result) - Stop browsing and return the result when the goal is achieved

Instructions:
- Analyze the current page (HTML and screenshot provided)
- Use CSS selectors only
- Think step by step to achieve the goal
- Be careful with selectors - make them specific to avoid errors
- If the page content is provided, analyze it before taking actions
- Report your findings and actions
- When the goal is achieved, use stop_browsing(result) with a summary of what was accomplished

Current goal: {GOAL}

Analyze the page and take appropriate actions."""

    target_url = "https://www.google.com"
    print(f"Opening URL: {target_url}")
    driver.get(target_url)
    
    # Initialize conversation history with first prompt
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_template)]
        )
    ]
    
    tools = get_browser_tools()
    config = types.GenerateContentConfig(
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY)
        )
    )
    
    step = 0
    while step < MAX_STEPS:
        step += 1
        print(f"\n{'='*60}")
        print(f"Step {step}/{MAX_STEPS}")
        print(f"{'='*60}")
        
        # Get current page state
        print("Capturing page state...")
        page_html, screenshot_file = get_page_state()
        print(f"Page HTML length: {len(page_html)} characters")
        
        # Add page context to the last user message for first iteration only
        # Add page context to the first user message for first iteration only
        if step == 1:
            contents[0].parts.extend([
                types.Part.from_text(text=f"\nCurrent page HTML:\n{page_html}"),
                types.Part.from_uri(
                    file_uri=screenshot_file.uri,
                    mime_type=screenshot_file.mime_type
                )
            ])
        
        # Send prompt with conversation history
        print("Sending prompt to Gemini...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )
        
        # Parse tool calls
        tool_calls = []
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    args = {}
                    if hasattr(function_call, 'args'):
                        if isinstance(function_call.args, dict):
                            args = function_call.args
                        elif hasattr(function_call.args, '__dict__'):
                            args = dict(function_call.args)
                    
                    tool_calls.append({
                        "name": function_call.name,
                        "arguments": args
                    })
        
        print(f"Received {len(tool_calls)} tool call(s)")
        
        # Append model's response to conversation history
        contents.append(response.candidates[0].content)
        
        # Execute tool calls
        if tool_calls:
            print("Executing tool calls...")
            for i, tool_call in enumerate(tool_calls, 1):
                print(f"Tool call {i}: {tool_call['name']} with args: {tool_call['arguments']}")
            
            # Check if stop_browsing is called
            stop_browsing_called = any(tc["name"] == "stop_browsing" for tc in tool_calls)
            
            results = execute_tool_calls(tool_calls)
            
            # Print results
            for i, result in enumerate(results, 1):
                if result["success"]:
                    print(f"✓ Tool call {i} ({result['function_name']}) succeeded")
                else:
                    print(f"✗ Tool call {i} ({result['function_name']}) failed: {result['error']}")
            
            # If stop_browsing was called, extract the result and break
            if stop_browsing_called:
                stop_result = None
                for tool_call, result in zip(tool_calls, results):
                    if tool_call["name"] == "stop_browsing" and result["success"]:
                        stop_result = result.get("result")
                        break
                
                print(f"\n{'='*60}")
                print("Browsing stopped by stop_browsing tool")
                print(f"{'='*60}")
                if stop_result:
                    print(f"Result: {stop_result}")
                print(f"Total steps executed: {step}")
                print(f"{'='*60}")
                break
            
            # Build function response parts
            function_response_parts = []
            for tool_call, result in zip(tool_calls, results):
                response_data = {
                    "result": result.get("result") if result["success"] else None,
                    "error": result.get("error") if not result["success"] else None,
                    "success": result["success"]
                }
                
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_call["name"],
                        response=response_data
                    )
                )
            # Add step progress information
            function_response_parts.append(
                types.Part.from_text(
                    text=f"\nStep {step}/{MAX_STEPS} completed. Remaining steps: {MAX_STEPS - step}. One the last step, use stop_browsing(result) to stop the browsing session."
                )
            )
            
            # Append function responses as empty user message
            contents.append(
                types.Content(
                    role="user",
                    parts=function_response_parts
                )
            )
            # wait before continuing
            time.sleep(1)
        
        else:
            # No tool calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"Model response: {part.text}")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print("Automation complete!")
    print(f"Total steps executed: {step}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
