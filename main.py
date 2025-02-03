from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

from dotenv import load_dotenv
from newsapi import NewsApiClient
import feedparser

load_dotenv()

class NewsAPI:
    def __init__(self, api_key):
        self.client = NewsApiClient(api_key=api_key)

    def get_top_headlines(self, **kwargs):
        return self.client.get_top_headlines(**kwargs)

class RSSFeed:
    def __init__(self, rss_url):
        self.rss_url = rss_url

    def fetch_feed(self):
        return feedparser.parse(self.rss_url)

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

""" news_agent = Agent(
    name="News Agent",
    description="This agent fetches and summarizes the latest news articles.",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[NewsAPI(api_key='336e7388f80e41bd9970c088b4a8b6e3'), RSSFeed(rss_url="https://rss.cnn.com/rss/edition.rss")],
    instructions=["Summarize the latest news articles and include sources."],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
) """

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

