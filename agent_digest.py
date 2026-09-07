import os
import json
import re
import feedparser
import resend
from datetime import datetime, timezone, timedelta
from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import VideosSearch

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

HISTORY_FILE = "sent_history.json"
RETENTION_DAYS = 30
RECENT_DAYS = 3


def parse_datetime_value(value):
    """Convert common RSS/YouTube timestamps into a UTC datetime."""
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, tuple) and len(value) >= 9:
        try:
            dt = datetime(*value[:6])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

        relative_match = re.match(
            r"(?i)^(?P<num>\d+)\s+(?P<unit>minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)\s+ago$",
            text,
        )
        if relative_match:
            num = int(relative_match.group("num"))
            unit = relative_match.group("unit").lower()
            units = {
                "minute": 1,
                "minutes": 1,
                "hour": 60,
                "hours": 60,
                "day": 24 * 60,
                "days": 24 * 60,
                "week": 7 * 24 * 60,
                "weeks": 7 * 24 * 60,
                "month": 30 * 24 * 60,
                "months": 30 * 24 * 60,
                "year": 365 * 24 * 60,
                "years": 365 * 24 * 60,
            }
            minutes_ago = num * units[unit]
            return datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)

        lowered = text.lower()
        if lowered in {"today", "yesterday"}:
            days = 0 if lowered == "today" else 1
            return datetime.now(timezone.utc) - timedelta(days=days)

    return None


def is_recent_enough(value, days: int = RECENT_DAYS) -> bool:
    """Return True when the value is within the last N days."""
    parsed_dt = parse_datetime_value(value)
    if parsed_dt is None:
        return False

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return parsed_dt >= cutoff


# ---------------------------------------------------------------------------
# 1. HISTORY DEDUPLICATION & 30-DAY RETENTION HELPERS
# ---------------------------------------------------------------------------

def load_sent_history() -> dict:
    """
    Loads sent history and automatically prunes any items older than 30 days.
    Returns a dictionary of { link_or_id: ISO_timestamp_str }.
    """
    if not os.path.exists(HISTORY_FILE):
        return {}

    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            
        # Backward compatibility for flat list format
        if isinstance(data, list):
            now_iso = datetime.now(timezone.utc).isoformat()
            data = {item: now_iso for item in data}

        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        active_history = {}

        for item_key, timestamp_str in data.items():
            try:
                item_time = datetime.fromisoformat(timestamp_str)
                if item_time >= cutoff:
                    active_history[item_key] = timestamp_str
            except Exception:
                continue

        return active_history
    except Exception:
        return {}


def save_sent_history(new_item_keys: list):
    """
    Appends newly sent items with current timestamp, prunes entries >30 days old,
    and writes the updated dictionary back to sent_history.json.
    """
    history = load_sent_history()
    now_iso = datetime.now(timezone.utc).isoformat()

    for item in new_item_keys:
        history[item] = now_iso

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ---------------------------------------------------------------------------
# 2. DEFINE AGENT TOOLS
# ---------------------------------------------------------------------------

@tool
def search_trending_youtube_videos(queries: list[str]) -> str:
    """
    Searches YouTube for new Data Engineering AI videos from the last 3 days.
    Filters out videos already sent within the last 30 days.
    """
    sent_history = load_sent_history()
    found_videos = []

    for query in queries:
        try:
            custom_search = VideosSearch(query, limit=5)
            results = custom_search.result().get('result', [])

            for video in results:
                video_id = video.get('id')
                if not video_id:
                    continue

                published_time = video.get('publishedTime', '')
                if not is_recent_enough(published_time, days=RECENT_DAYS):
                    continue

                video_url = f"https://www.youtube.com/watch?v={video_id}"

                # Skip items already sent in the last 30 days
                if video_url in sent_history or video_id in sent_history:
                    continue

                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = " ".join([t['text'] for t in transcript_list[:40]])
                except Exception:
                    transcript_text = "Transcript unavailable."

                found_videos.append({
                    "id": video_id,
                    "title": video.get('title'),
                    "link": video_url,
                    "channel": video.get('channel', {}).get('name', 'Unknown Channel'),
                    "views": video.get('viewCount', {}).get('short', 'N/A'),
                    "publish_time": published_time,
                    "transcript_snippet": transcript_text
                })
        except Exception:
            continue

    return json.dumps(found_videos, indent=2)


@tool
def fetch_rss_updates(rss_urls: list[str]) -> str:
    """
    Fetches only RSS entries from the last 3 days.
    Filters out articles already sent within the last 30 days.
    """
    sent_history = load_sent_history()
    fetched_data = []

    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            article_link = getattr(entry, "link", "")
            if not article_link:
                continue

            published_value = getattr(entry, "published", None)
            if not published_value:
                published_value = getattr(entry, "published_parsed", None)
            if not published_value:
                published_value = getattr(entry, "updated", None)
            if not published_value:
                published_value = getattr(entry, "updated_parsed", None)

            if not is_recent_enough(published_value, days=RECENT_DAYS):
                continue

            if article_link in sent_history:
                continue

            fetched_data.append({
                "title": getattr(entry, "title", "No Title"),
                "link": article_link,
                "summary": getattr(entry, "summary", ""),
                "published": published_value
            })

    return json.dumps(fetched_data, indent=2)


@tool
def send_email_digest(to_email: str, subject: str, html_content: str, sent_item_links: list[str]) -> str:
    """
    Sends formatted HTML email digest via Resend API and records sent links in history.
    """
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
        
        # Save newly sent items and prune entries older than 30 days
        save_sent_history(sent_item_links)
            
        return "Email sent successfully and 30-day history updated."
    except Exception as e:
        return f"Failed to send email: {str(e)}"

# ---------------------------------------------------------------------------
# 3. RUNNER PIPELINE WITH DIRECT TOOL BINDING
# ---------------------------------------------------------------------------

def run_agent_pipeline():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    # Matches your exact GitHub Action secret variable spelling
    recipient_email = os.environ.get("RECEPEINT_EMAIL")

    if not gemini_api_key or not recipient_email:
        raise ValueError("Missing GEMINI_API_KEY or RECEPEINT_EMAIL environment variables.")

    tools = [search_trending_youtube_videos, fetch_rss_updates, send_email_digest]
    tools_by_name = {t.name: t for t in tools}

    # Bind tools directly to Gemini model using gemini-3.6-flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=gemini_api_key,
        temperature=0.2
    ).bind_tools(tools)

    system_instruction = """
    You are an autonomous Senior Data Engineering AI Agent.
    Your goal is to find new trending YouTube videos and RSS news published recently 
    focused on:
    - Snowflake Cortex AI
    - Databricks LTAP / Lakehouse developments
    - AI-driven Data Governance & Quality
    - Data Ingestion & Real-time Pipelines
    - Trending Data Engineering architectural updates

    Workflow:
    1. Call `search_trending_youtube_videos` using relevant search terms.
    2. Call `fetch_rss_updates` for core blog updates.
    3. Evaluate retrieved items through a Data Engineering lens.
    4. Construct a clean HTML digest containing high-value items (Score 7/10+):
       - Video / Article Title & Direct Link
       - Channel Name / Views / Publish Date
       - **What's New in There:** (Key features or announcements)
       - **Why You Need to Watch/Read:** (Impact on engineering pipelines, performance, or costs)
    5. Pass all the links included in your email into the `sent_item_links` parameter of `send_email_digest`.
    6. If no new high-quality items are found (all retrieved items were sent recently or are irrelevant), 
       send a lightweight email stating: "<p>No new high-signal Data Engineering AI updates found today.</p>"
    """

    user_prompt = f"""
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
      "https://blog.langchain.dev/rss/",
      "https://www.snowflake.com/blog/feed/",
      "https://www.dataengineeringweekly.com/feed/",
      "https://www.databricks.com/blog/rss.xml",
      "https://netflixtechblog.com/feed",
      "https://huggingface.co/blog/feed.xml
    ]

    Filter for high-signal updates and send the daily digest email to "{recipient_email}".
    """

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    messages.append(response)

    # Tool Execution Loop
    while response.tool_calls:
        for tool_call in response.tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            
            messages.append({
                "role": "tool",
                "content": str(tool_output),
                "tool_call_id": tool_call["id"]
            })
            
        response = llm.invoke(messages)
        messages.append(response)

    print("Agent pipeline completed execution successfully.")

if __name__ == "__main__":
    run_agent_pipeline()