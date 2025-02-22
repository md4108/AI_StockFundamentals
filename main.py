from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

from dotenv import load_dotenv
# from newsapi import NewsApiClient
# import feedparser
from langchain.tools import Tool
from newsapi import NewsApiClient
import feedparser

load_dotenv()

class NewsAPIWrapper:  # Wrapper class to make it a Langchain Tool
    def __init__(self, api_key):
        self.client = NewsApiClient(api_key=api_key)

    def get_top_headlines(self, query: str) -> str: # Added query parameter for tool usage
        """Fetch top news headlines based on a query. """
        try:
            response = self.client.get_top_headlines(q=query, language='en') # Added q parameter for query
            articles = response.get('articles', [])
            if articles:
                headlines = "\n".join([f"- {article['title']}: {article['description']}" for article in articles])
                return headlines
            else:
                return "No top headlines found for the given query. "
        except Exception as e:
            return f"Error fetching news: {e}"

class RSSFeedWrapper:
    def __init__(self, rss_url):
        self.rss_url = rss_url

    def fetch_feed(self, query: str) -> str:  # Added query parameter for tool usage
        """Fetch and summarize an RSS feed based on a query."""
        try:
            feed = feedparser.parse(self.rss_url)
            entries = feed.entries
            relevant_entries = [entry for entry in entries if query.lower() in entry.title.lower() or query.lower() in entry.description.lower()]

            if relevant_entries:
                summary = "\n".join([f"- {entry.title}: {entry.description}" for entry in relevant_entries])
                return summary
            else:
                return "No relevant entries found in the RSS feed for the given query."
        except Exception as e:
            return f"Error fetching RSS feed: {e}"

news_api = NewsAPIWrapper(api_key='336e7388f80e41bd9970c088b4a8b6e3')
rss_feed = RSSFeedWrapper(rss_url="https://rss.cnn.com/rss/edition.rss")

news_api_tool = Tool(
    name="NewsAPI",
    func=news_api.get_top_headlines,
    description="Useful for fetching the latest news headlines.  Input should be a search query related to a topic or company."
)

rss_feed_tool = Tool(
    name="RSSFeed",
    func=rss_feed.fetch_feed,
    description="Useful for fetching information from an RSS feed.  Input should be a search query related to a topic or company."
)

web_search_agent = Agent(
    name = "Web Agent",
    description = "This is the agent for searching content from the web",
    model = Groq(id="deepseek-r1-distill-llama-70b"),
    tools  = [DuckDuckGo()],
    instructions = "Always include the sources",
    show_tool_calls = True,
    markdown = True,
    debug_mode=True
)

#web_search_agent.print_response("What is the capital of Nepal?", stream=True)

finance_agent = Agent(
    name="Finance Agent",
    description = "Your task is to find the finance information",
    model = Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode = True
)
#finance_agent.print_response("Summarize analyst recommendations for NVDA", stream=True)

news_agent = Agent(
    name="News Agent",
    description="This agent fetches and summarizes the latest news articles.",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[news_api_tool, rss_feed_tool],
    instructions=["Provide latest news of the mentioned tickers in the query"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
) 

agent_team = Agent(
    team=[web_search_agent, finance_agent],
    model = Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode =True
)


response = agent_team.print_response("Can you list top 2 stocks listed on Indian stock market and their Ticker names, then summarize analyst recommendations and provide latest news about them", stream=True)

#response = agent_team.respond_directly("Can you list top 2 stocks listed on US stock market and their Ticker names, and then summarize analyst recommendations")
if response:
    with open("output.txt", "w") as file:
        file.write(response)
else:
    print("No response received from agent_team.")

