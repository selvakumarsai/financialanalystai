import streamlit as st
import os
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent


# Suppress warnings from OpenBB
# import warnings
# warnings.filterwarnings("ignore")

# --- Streamlit UI for API Key and PAT input ---
st.set_page_config(page_title="Financial Analyst AI", layout="centered")

st.title("📈 Financial Analyst AI")

openai_api_key = st.secrets["openai_api_key"]
openbb_pat = st.secrets["openbb_pat"]

if not openai_api_key or not openbb_pat:
    st.warning("Please configure both your OpenAI API Key and OpenBB PAT as secrets to proceed.")
    st.stop()


# Initialize OpenBB
try:
    from openbb import obb
    obb.account.login(pat=openbb_pat)
    st.success("OpenBB and OpenAI initialized successfully!")
except Exception as e:
    st.error(f"Error initializing OpenBB or OpenAI. Please check your keys. Error: {e}")
    st.stop()

# --- Define the tools ---

@tool
def get_stock_ticker_symbol(stock_name: str) -> str:
    """Get the symbol, name and CIK for any publicly traded company"""
    try:
        res = obb.equity.search(stock_name, provider="sec")
        if res.to_df().empty:
            return f"No ticker symbol found for '{stock_name}'. Please check the spelling or try a different name."
        stock_ticker_details = res.to_df().to_markdown()
        output = f"Here are the details of the company and its stock ticker symbol:\n\n{stock_ticker_details}"
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

        output = (f"Here are the stock price metrics and data for the stock ticker symbol {stock_ticker}:\n\n"
                  f"Price Quote Metrics:\n\n{price_quote}\n\n"
                  f"Price Performance Metrics:\n\n{price_performance}\n\n"
                  f"Price Historical Data:\n\n{price_historical}")
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

        output = (f"Here are the fundamental indicator metrics and data for the stock ticker symbol {stock_ticker}:\n\n"
                  f"Fundamental Ratios:\n\n{fundamental_ratios}\n\n"
                  f"Fundamental Metrics:\n\n{fundamental_metrics}")
        return output
    except Exception as e:
        return f"Error fetching fundamental indicator metrics for {stock_ticker}: {e}"

@tool
def get_stock_news(stock_ticker: str) -> str:
    """Get news article headlines for a specific stock ticker"""
    try:
        end_date = datetime.now()
        start_date = (end_date - timedelta(days=45)).strftime("%Y-%m-%d")
        res_news = obb.news.company(symbol=stock_ticker, start_date=start_date, provider='tmx', limit=50)
        
        if res_news.to_df().empty:
            return f"No recent news found for {stock_ticker}."
        
        news = res_news.to_df()[['symbols', 'title']].to_markdown()
        output = f"Here are the recent news headlines for the stock ticker symbol {stock_ticker}:\n\n{news}"
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

        output = (f"Here's some detailed information of the stock market which includes most actively traded stocks, gainers and losers:\n\n"
                  f"Most actively traded stocks:\n\n{most_active_stocks}\n\n"
                  f"Top price gainers:\n\n{price_gainers}\n\n"
                  f"Top price losers:\n\n{price_losers}")
        return output
    except Exception as e:
        return f"Error fetching general market data: {e}"

# --- Agent Setup ---
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [get_stock_ticker_symbol,
         get_stock_price_metrics,
         get_stock_fundamental_indicator_metrics,
         get_stock_news,
         get_general_market_data]

AGENT_PREFIX = """Role: You are an AI stock market assistant tasked with providing investors
with up-to-date, detailed information on individual stocks or advice based on general market data.

Objective: Assist data-driven stock market investors by giving accurate,
complete, but concise information relevant to their questions about individual
stocks or general advice on useful stocks based on general market data and trends.

Capabilities: You are given a number of tools as functions. Use as many tools
as needed to ensure all information provided is timely, accurate, concise,
relevant, and responsive to the user's query.

Starting Flow:
Input validation. Determine if the input is asking about a specific company
or stock ticker (Flow 2). If not, check if they are asking for general advice on potentially useful stocks
based on current market data (Flow 1). Otherwise, respond in a friendly, positive, professional tone
that you don't have information to answer as you can only provide financial advice based on market data.
For each of the flows related to valid questions use the following instructions:

Flow 1:
A. Market Analysis: If the query is valid and the user wants to get general advice on the market
or stocks worth looking into for investing, leverage the general market data tool to get relevant data.

Flow 2:
A. Symbol extraction. If the query is valid and is related to a specific company or companies,
extract the company name or ticker symbol from the question.
If a company name is given, look up the ticker symbol using a tool.
If the ticker symbol is not found based on the company, try to
correct the spelling and try again, like changing "microsfot" to "microsoft",
or broadening the search, like changing "southwest airlines" to a shorter variation
like "southwest" and increasing "limit" to 10 or more. If the company or ticker is
still unclear based on the question or conversation so far, and the results of the
symbol lookup, then ask the user to clarify which company or ticker.

B. Information retrieval. Determine what data the user is seeking on the symbol
identified. Use the appropriate tools to fetch the requested information. Only use
data obtained from the tools. You may use multiple tools in a sequence. For instance,
first determine the company's symbol, then retrieve price data using the symbol
and fundamental indicator data etc. For specific queries only retrieve data using the most relevant tool.
If detailed analysis is needed, you can call multiple tools to retrieve data first.

Response Generation Flow:
Compose Response. Analyze the retrieved data carefully and provide a comprehensive answer to the user in a clear and concise format,
in a friendly professional tone, emphasizing the data retrieved.
If the user asks for recommendations you can give some recommendations
but emphasize the user to do their own research before investing.
When generating the final response in markdown,
if there are special characters in the text, such as the dollar symbol,
ensure they are escaped properly for correct rendering e.g $25.5 should become \$25.5

Example Interaction:
User asks: "What is the PE ratio for Eli Lilly?"
Chatbot recognizes 'Eli Lilly' as a company name.
Chatbot uses symbol lookup to find the ticker for Eli Lilly, returning LLY.
Chatbot retrieves the PE ratio using the proper function with symbol LLY.
Chatbot responds: "The PE ratio for Eli Lilly (symbol: LLY) as of May 12, 2024 is 30."

Check carefully and only call the tools which are specifically named below.
Only use data obtained from these tools.
"""

SYS_PROMPT = SystemMessage(content=AGENT_PREFIX)

financial_analyst = create_react_agent(model=chatgpt,
                                       tools=tools,
                                       state_modifier=SYS_PROMPT)

# --- Streamlit App Logic ---
st.subheader("Ask your financial questions!")
user_query = st.text_area("Enter your query here:", "Tell me how is Nvidia doing right now as a company and could I potentially invest in it?")

if st.button("Get Analysis"):
    if user_query:
        with st.spinner("Analyzing your request..."):
            try:
                # The stream method returns an iterator of events
                # We need to iterate through it to get the final message
                final_response = ""
                for event in financial_analyst.stream(
                    {"messages": [HumanMessage(content=user_query)]},
                    stream_mode='values'
                ):
                    # The last message in the event is the one we want to display
                    if event["messages"]:
                        final_response = event["messages"][-1].content

                if final_response:
                    st.markdown("### Analysis Result:")
                    st.markdown(final_response)
                else:
                    st.error("Could not retrieve a response. Please try again or refine your query.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("Please enter a query to get analysis.")

