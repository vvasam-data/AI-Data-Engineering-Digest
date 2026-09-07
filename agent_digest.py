import os
import json
import feedparser
import resend
from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import VideosSearch
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. DEFINE TOOLS FOR THE AGENT HARNESS
# ---------------------------------------------------------------------------

@tool
def search_trending_youtube_videos(queries: list[str]) -> str:
    """
    Searches YouTube for recent videos matching Data Engineering AI topics 
    (Snowflake Cortex, Databricks LTAP, Data Governance, Data Ingestion).
    Extracts video metadata and transcripts for evaluation.
    """
    found_videos = []
    
    for query in queries:
        try:
            # Search YouTube using VideosSearch
            custom_search = VideosSearch(query, limit=5)
            results = custom_search.result().get('result', [])
            
            for video in results:
                video_id = video.get('id')
                if not video_id:
                    continue
                    
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Fetch transcript snippet
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = " ".join([t['text'] for t in transcript_list[:40]])
                except Exception:
                    transcript_text = "Transcript unavailable."

                found_videos.append({
                    "title": video.get('title'),
                    "link": video_url,
                    "channel": video.get('channel', {}).get('name', 'Unknown Channel'),
                    "views": video.get('viewCount', {}).get('short', 'N/A'),
                    "publish_time": video.get('publishedTime', 'N/A'),
                    "transcript_snippet": transcript_text
                })
        except Exception as e:
            continue

    return json.dumps(found_videos, indent=2)


@tool
def fetch_rss_updates(rss_urls: list[str]) -> str:
    """Fetches latest entries from specified RSS feeds."""
    fetched_data = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            fetched_data.append({
                "title": entry.title,
                "link": entry.link,
                "summary": getattr(entry, "summary", "")
            })
    return json.dumps(fetched_data, indent=2)


@tool
def send_email_digest(to_email: str, subject: str, html_content: str) -> str:
    """Sends formatted HTML email digest via Resend API."""
    resend.api_key = os.environ.get("RESEND_API_KEY")
    if not resend.api_key:
        return "Error: RESEND_API_KEY environment variable not configured."

    try:
        resend.Emails.send({
            "from": "AI Agent <onboarding@resend.dev>",
            "to": to_email,
            "subject": subject,
            "html": html_content
        })
        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {str(e)}"

# ---------------------------------------------------------------------------
# 2. CONFIGURE AGENT HARNESS & EXECUTION
# ---------------------------------------------------------------------------

def run_agent_pipeline():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")

    if not gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")
    if not recipient_email:
        raise ValueError("Missing RECIPIENT_EMAIL environment variable.")

    # Initialize LLM with Google Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.2
    )

    tools = [search_trending_youtube_videos, fetch_rss_updates, send_email_digest]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an autonomous Senior Data Engineering AI Agent.
        Your goal is to find trending YouTube videos and RSS news published recently (last 3 days) 
        focused on:
        - Snowflake Cortex AI
        - Databricks LTAP / Lakehouse developments
        - AI-driven Data Governance & Quality
        - Data Ingestion & Real-time Pipelines
        - Trending Data Engineering architectural updates

        Workflow:
        1. Call `search_trending_youtube_videos` using relevant search terms.
        2. Call `fetch_rss_updates` for core blog updates.
        3. Evaluate content through a Data Engineering lens.
        4. For high-value items (Score 7/10 or higher), construct an HTML digest containing:
           - Video / Article Title & Direct Link
           - Channel Name / Views / Publish Date
           - **What's New in There:** (Key features or announcements)
           - **Why You Need to Watch/Read:** (Impact on engineering pipelines, performance, or costs)
        5. Dispatch the HTML digest using `send_email_digest`.
        """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    input_payload = {
        "input": f"""
        Execute YouTube searches for:
        [
          "Snowflake Cortex AI",
          "Databricks LTAP",
          "AI Data Governance",
          "Data Ingestion AI Data Engineering"
        ]

        Fetch RSS feeds from:
        [
          "https://rss.arxiv.org/rss/cs.DB",
          "https://blog.langchain.dev/rss/"
        ]

        Filter for high-signal updates from the last 3 days and send the email digest to "{recipient_email}".
        """
    }

    agent_executor.invoke(input_payload)

if __name__ == "__main__":
    run_agent_pipeline()