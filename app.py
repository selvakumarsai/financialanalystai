import streamlit as st
import os
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, Literal, TypedDict
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.graph.message import add_messages
# Removed from getpass and IPython.display imports

# --- Streamlit UI for API Key and PAT input ---
st.set_page_config(page_title="Financial Analyst AI", layout="centered")

st.title("📈 Financial Analyst AI")

# Retrieve secrets from Streamlit's secrets management
openai_api_key = st.secrets.get("openai_api_key")
openbb_pat = st.secrets.get("openbb_pat")
tavily_api_key = st.secrets.get("tavily_api_key")

if not openai_api_key:
    st.warning("Please configure your OpenAI API Key as a secret (openai_api_key).")
    st.stop()
if not openbb_pat:
    st.warning("Please configure your OpenBB Personal Access Token as a secret (openbb_pat).")
    st.stop()
if not tavily_api_key:
    st.warning("Please configure your Tavily Search API Key as a secret (tavily_api_key).")
    st.stop()


# Set environment variables for the current session (crucial for OpenBB and LangChain)
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['TAVILY_API_KEY'] = tavily_api_key


# --- Configure OpenBB writable directories for Streamlit Community Cloud ---
# This is crucial for Permission Denied errors.
# Use a subdirectory within the app's root (/app/ on Streamlit Community Cloud)
# which is writable. The /tmp directory is also usually writable.
try:
    openbb_data_dir = os.path.join(os.getcwd(), ".openbb_data")
    openbb_log_dir = os.path.join(os.getcwd(), ".openbb_logs")

    os.makedirs(openbb_data_dir, exist_ok=True)
    os.makedirs(openbb_log_dir, exist_ok=True)

    os.environ["OPENBB_USER_DATA_DIRECTORY"] = openbb_data_dir
    os.environ["OPENBB_LOG_DIRECTORY"] = openbb_log_dir

    st.info(f"OpenBB User Data Directory: {os.environ['OPENBB_USER_DATA_DIRECTORY']}")
    st.info(f"OpenBB Log Directory: {os.environ['OPENBB_LOG_DIRECTORY']}")

except Exception as e:
    st.error(f"Error setting OpenBB environment variables or creating directories: {e}")
    st.stop()


# Initialize OpenBB
try:
    from openbb import obb
    # The 'login' step can take a moment, especially if it's the first time
    with st.spinner("Logging into OpenBB account..."):
        obb.account.login(pat=openbb_pat)
    st.success("OpenBB and OpenAI initialized successfully!")
except Exception as e:
    st.error(f"Error initializing OpenBB or OpenAI. Please check your keys and OpenBB setup. Error: {e}")
    st.stop()

# --- Initialize Tools ---
tavily_search = TavilySearchAPIWrapper()
repl = PythonREPL()

@tool
def search_web(query: str, num_results=10) -> list:
    """Search the web for a query. Userful for general information or general news"""
    results = tavily_search.raw_results(query=query,
                                        max_results=num_results,
                                        search_depth='advanced',
                                        include_answer=False,
                                        include_raw_content=True)
    # Return only snippets or relevant parts to avoid overwhelming context
    # You might want to refine this to only return the 'content' or 'snippet' of each result
    return [{"title": r['title'], "snippet": r['content']} for r in results if 'content' in r and 'title' in r][:num_results]

@tool
def get_stock_ticker_symbol(stock_name: str) -> str:
    """Get the symbol, name and CIK for any publicly traded company"""
    try:
        res = obb.equity.search(stock_name, provider="sec")
        if res.to_df().empty:
            return f"No ticker symbol found for '{stock_name}'. Please check the spelling or try a different name."
        stock_ticker_details = res.to_df().to_markdown()
        output = """Here are the details of the company and its stock ticker symbol:\n\n""" + stock_ticker_details
        return output
    except Exception as e:
        return f"Error fetching stock ticker symbol for {stock_name}: {e}"

@tool
def get_stock_price_metrics(stock_ticker: str) -> str:
    """Get historical stock price data, stock price quote and price performance data
       like price changes for a specific stock ticker"""
    try:
        res_quote = obb.equity.price.quote(stock_ticker, provider='cboe')
        price_quote = res_quote.to_df().to_markdown() if not res_quote.to_df().empty else "No price quote available."

        res_performance = obb.equity.price.performance(symbol=stock_ticker, provider='finviz')
        price_performance = res_performance.to_df().to_markdown() if not res_performance.to_df().empty else "No price performance available."

        end_date = datetime.now()
        start_date = (end_date - timedelta(days=365)).strftime("%Y-%m-%d")
        res_historical = obb.equity.price.historical(symbol=stock_ticker, start_date=start_date,
                                                      interval='1d', provider='yfinance')
        price_historical = res_historical.to_df().to_markdown() if not res_historical.to_df().empty else "No historical price data available."

        output = ("""Here are the stock price metrics and data for the stock ticker symbol """ + stock_ticker + """: \n\n""" +
                  "Price Quote Metrics:\n\n" + price_quote +
                  "\n\nPrice Performance Metrics:\n\n" + price_performance +
                  "\n\nPrice Historical Data:\n\n" + price_historical)
        return output
    except Exception as e:
        return f"Error fetching stock price metrics for {stock_ticker}: {e}"

@tool
def get_stock_fundamental_indicator_metrics(stock_ticker: str) -> str:
    """Get fundamental indicator metrics for a specific stock ticker"""
    try:
        res_ratios = obb.equity.fundamental.ratios(symbol=stock_ticker, period='annual',
                                                    limit=10, provider='fmp')
        fundamental_ratios = res_ratios.to_df().to_markdown() if not res_ratios.to_df().empty else "No fundamental ratios available."

        res_metrics = obb.equity.fundamental.metrics(symbol=stock_ticker, period='annual',
                                                    limit=10, provider='yfinance')
        fundamental_metrics = res_metrics.to_df().to_markdown() if not res_metrics.to_df().empty else "No fundamental metrics available."

        output = ("""Here are the fundamental indicator metrics and data for the stock ticker symbol """ + stock_ticker + """: \n\n""" +
                  "Fundamental Ratios:\n\n" + fundamental_ratios +
                  "\n\nFundamental Metrics:\n\n" + fundamental_metrics)
        return output
    except Exception as e:
        return f"Error fetching fundamental indicator metrics for {stock_ticker}: {e}"

@tool
def get_stock_news(stock_ticker: str) -> str:
    """Get news article headlines for a specific stock ticker"""
    try:
        end_date = datetime.now()
        start_date = (end_date - timedelta(days=45)).strftime("%Y-%m-%d")
        res = obb.news.company(symbol=stock_ticker, start_date=start_date, provider='tmx', limit=50)
        
        if res.to_df().empty:
            return f"No recent news found for {stock_ticker}."
        
        news_df = res.to_df()
        # Filter for relevant columns
        news_df = news_df[['symbols', 'title']]
        news = news_df.to_markdown()

        output = ("""Here are the recent news headlines for the stock ticker symbol """ + stock_ticker + """: \n\n""" + news)
        return output
    except Exception as e:
        return f"Error fetching stock news for {stock_ticker}: {e}"

@tool
def get_general_market_data() -> str:
    """Get general data and indicators for the whole stock market including,
       most actively traded stocks based on volume, top price gainers and top price losers.
       Useful when you want an overview of the market and what stocks to look at."""
    try:
        res_active = obb.equity.discovery.active(sort='desc', provider='yfinance', limit=15)
        most_active_stocks = res_active.to_df().to_markdown() if not res_active.to_df().empty else "No actively traded stocks data available."

        res_gainers = obb.equity.discovery.gainers(sort='desc', provider='yfinance', limit=15)
        price_gainers = res_gainers.to_df().to_markdown() if not res_gainers.to_df().empty else "No top gainers data available."

        res_losers = obb.equity.discovery.losers(sort='desc', provider='yfinance', limit=15)
        price_losers = res_losers.to_df().to_markdown() if not res_losers.to_df().empty else "No top losers data available."

        output = ("""Here's some detailed information of the stock market which includes most actively traded stocks, gainers and losers:\n\n""" +
                  "Most actively traded stocks:\n\n" + most_active_stocks +
                  "\n\nTop price gainers:\n\n" + price_gainers +
                  "\n\nTop price losers:\n\n" + price_losers)
        return output
    except Exception as e:
        return f"Error fetching general market data: {e}"

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute code, generate charts."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    Note: For plotting, the Python REPL cannot directly display interactive charts in Streamlit.
    It can generate data that you then manually display in Streamlit, or return base64 encoded images.
    For this application, assume it will return text output."""
    try:
        # The REPL will execute the code. If plotting, it will only generate text output.
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nCODE OUTPUT:\n {result}"
    return result_str

# --- Agent Setup ---
# No change needed for this part from your original script, except making sure ChatOpenAI uses os.environ for API key
llm = ChatOpenAI(model="gpt-4o", temperature=0)

members = ["researcher", "coder"]

SUPERVISOR_AGENT_PROMPT = f"""You are a supervisor tasked with managing a conversation between the following workers:
                              {members}.

                              Given the following user request, respond with the worker to act next.
                              Each worker will perform a task and respond with their results and status.
                              Analyze the results carefully and decide which worker to call next accordingly.
                              Remember researcher agent can search for information and coder agent can code.
                              When finished, respond with FINISH."""


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["researcher", "coder", "FINISH"]

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

def supervisor_node(state: State) -> Command[Literal["researcher", "coder", "__end__"]]:
    messages = [{"role": "system", "content": SUPERVISOR_AGENT_PROMPT},] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

# Create Financial Researcher Sub-Agent
research_agent = create_react_agent(
    llm, tools=[search_web,
            get_stock_ticker_symbol,
            get_stock_price_metrics,
            get_stock_fundamental_indicator_metrics,
            get_stock_news,
            get_general_market_data], state_modifier="""You are a financial researcher who excels in searching the web and financial platforms and analyzing the data.
                                                        DO NOT do any math or coding.
                                                        Once your task is done report back to the supervisor."""
)

def research_node(state: State) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )

code_agent = create_react_agent(llm, tools=[python_repl_tool], state_modifier="""You are a coder who can write and run python code and also visualize charts and graphs.
                                                                                 Only extract the most relevant data related to the question before running code or creating graphs.
                                                                                 Once your task is done report back to the supervisor.""")

# create node function for coder sub-agent
def code_node(state: State) -> Command[Literal["supervisor"]]:
    result = code_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="supervisor",
    )

builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)
graph = builder.compile()

# --- Streamlit App Logic (Interactive Execution) ---
st.subheader("Ask your financial questions!")
user_query = st.text_area(
    "Enter your query here:",
    "Get the stock price details of nvidia and intel and display it as a line chart in the same plot comparing the trend"
)

if st.button("Get Analysis"):
    if user_query:
        st.write("Initializing agent...")
        final_response_content = ""
        with st.spinner("Analyzing your request..."):
            try:
                # The stream method returns an iterator of events
                for event in graph.stream(
                    {"messages": [("user", user_query)]},
                    {"recursion_limit": 150},
                    stream_mode='values'
                ):
                    # Iterate through messages in the event and display them
                    for message in event["messages"]:
                        # Display intermediate steps or final result
                        if message.content: # Ensure message has content
                            if message.name:
                                st.write(f"**{message.name.capitalize()}:** {message.content}")
                            else:
                                st.write(message.content) # Final response from user role
                            final_response_content = message.content # Keep track of the last message

                if final_response_content:
                    st.markdown("### Final Analysis Result:")
                    st.markdown(final_response_content) # Display the final response
                else:
                    st.error("Could not retrieve a response. Please try again or refine your query.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e) # Show full exception traceback for debugging
    else:
        st.warning("Please enter a query to get analysis.")
