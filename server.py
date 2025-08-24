"""
Simple Flask server to handle chatbot queries via the OpenAI API.

This script defines a single `/chat` endpoint that accepts a JSON POST
payload with a `question` field.  It then sends that question, along
with a predefined knowledge base about Dimitris Dakos, to the
OpenAI ChatCompletion API and returns the model's answer.

To run this server locally you need to install the required
dependencies and set an environment variable called
`OPENAI_API_KEY` with your OpenAI API key.  For example:

    pip install flask openai
    export OPENAI_API_KEY="sk-..."
    python server.py

You can then send POST requests to http://localhost:5000/chat with
JSON payloads like {"question": "What are his main achievements?"}.

Please note: Because this environment does not have access to the
internet or your API key, the API call cannot be tested here.  You
must run the server on your own machine or a cloud environment with
network access and a valid API key.
"""

import os
from flask import Flask, request, jsonify
import openai


app = Flask(__name__)


def get_knowledge_base() -> str:
    """Return a single string containing the knowledge base.

    In a real application you might read this from a file or database.
    Here we hard‑code it based on Dimitris Dakos’s CV and LinkedIn
    information.  The knowledge base should be concise but contain
    enough detail for the model to answer questions.  Ensure that
    everything here is factual and comes from the user’s documents.
    """
    return (
        "Dimitris Dakos is a seasoned supply‑chain executive.\n"
        "He is currently Director of Warehousing at OB Streem (since Sept 2024).\n"
        "In this role he manages multiple sites totalling about 239 k m² of property and 90 k m² of buildings,\n"
        "leads around 130 people, handles budgets of 9–12 M €, opens new facilities, buys equipment and coaches his team.\n"
        "He has improved cost–service performance and margins by ~19 %.\n\n"
        "From 2008–2024 he was Head of Distribution Center for AB Vassilopoulos (Ahold Delhaize).\n"
        "There he ran a 33 k m² facility on a 112 k m² site, serving 133 stores with about 180 staff, \n"
        "and managed budgets of 7–10 M €.  Achievements included +5 % picking productivity, \n"
        "–7 % errors/shrinkage, –10 % yard wait time, +15 % reception capacity, a workflow automation saving 10 % workhours, \n"
        "an initiative for frozen products saving €60 k annually, material‑flow automation saving two full‑time equivalents,\n"
        "and scaling the centre from serving 34 stores to 133.\n\n"
        "Earlier he built the logistics department at Philkeram Johnson (Norcros Group).  \n"
        "He managed purchasing, demand planning, production scheduling and warehousing with a 6 M € budget and 30 people.\n"
        "A stock‑management plan generated €2.2 M in cash flow over three years; an order‑management system increased service levels by 9 %\n"
        "and lowered inventory carrying costs by 17 %, cut wait time for backorder info by 50 % and supported a fivefold increase in SKUs.\n\n"
        "Dimitris holds an MSc in Logistics Management (City College/University of Sheffield, 2003–2005), \n"
        "an MSc in Marketing Management (Aston University, 1997–1998), and a BA in Business Studies (University of Sheffield, 1994–1997).\n"
        "He is PMP certified and has completed MITx and HarvardX courses in manufacturing systems, supply‑chain analytics, agile and AI/ML.\n"
        "He is a board member of the Hellenic Society of Logistics.  His interests include supply‑chain strategy, project management, agile,\n"
        "continuous improvement, leadership development, emerging technologies (AI, ML, data science, blockchain/NFTs), education, \n"
        "diversity and inclusion, sustainability and climate action.\n"
    )


def query_openai(messages):
    """Send a chat completion request to OpenAI and return the reply text."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message["content"].strip()


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    question = data.get("question", "").strip() if data else ""
    if not question:
        return jsonify({"error": "No question provided"}), 400
    kb = get_knowledge_base()
    # Build a prompt: system message instructs the model to use only the knowledge base.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who answers questions about Dimitris Dakos based solely on the provided knowledge base.\n"
                "Do not invent facts; if the answer is not in the knowledge base, respond that you don't know.\n"
                f"Knowledge base:\n{kb}"
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        answer = query_openai(messages)
        return jsonify({"answer": answer})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)