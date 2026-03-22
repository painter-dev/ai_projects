import os
import json
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from anthropic import Anthropic
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
)

load_dotenv()

app = Flask(__name__)
PROPERTY_ID = os.environ["GA_PROPERTY_ID"]
claude = Anthropic()
ga_client = BetaAnalyticsDataClient()

tools = [
    {
        "name": "run_ga_report",
        "description": (
            "Run a Google Analytics report. Use this to answer questions about "
            "website traffic, sessions, users, pageviews, bounce rate, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "GA4 dimensions e.g. 'date', 'pagePath', 'sessionSource', 'country', 'deviceCategory'",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "GA4 metrics e.g. 'sessions', 'activeUsers', 'screenPageViews', 'bounceRate'",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD or relative like '7daysAgo', '30daysAgo', 'yesterday'",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format or 'today'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 10)",
                },
            },
            "required": ["dimensions", "metrics", "start_date", "end_date"],
        },
    }
]


def run_ga_report(dimensions, metrics, start_date, end_date, limit=10):
    request = RunReportRequest(
        property=f"properties/{PROPERTY_ID}",
        dimensions=[Dimension(name=d) for d in dimensions],
        metrics=[Metric(name=m) for m in metrics],
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        limit=limit,
    )
    response = ga_client.run_report(request)
    headers = [d.name for d in response.dimension_headers] + [
        m.name for m in response.metric_headers
    ]
    rows = []
    for row in response.rows:
        values = [dv.value for dv in row.dimension_values] + [
            mv.value for mv in row.metric_values
        ]
        rows.append(dict(zip(headers, values)))
    return {"headers": headers, "rows": rows, "row_count": len(rows)}


def run_agent(messages):
    while True:
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=(
                "You are a helpful Google Analytics analyst. "
                "Use the run_ga_report tool to fetch data and answer questions clearly. "
                "Summarize insights in plain language — no jargon."
            ),
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_ga_report(**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            messages.append({"role": "user", "content": tool_results})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    try:
        reply = run_agent(messages)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
