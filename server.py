"""
Flask API for a knowledge‑based chatbot using the OpenAI v1 client.

This server exposes a single `/chat` endpoint which accepts JSON POST
requests with a `question` field.  It uses a concise knowledge base
derived from Dimitris Dakos’s CV and LinkedIn posts, and instructs the
OpenAI model to answer strictly from this knowledge.  When the
knowledge base does not cover the question, the model is told to
respond that it does not have the information.

The server also supports CORS (Cross‑Origin Resource Sharing) so that
the API can be safely called from a GitHub Pages front‑end.  You
must set an `OPENAI_API_KEY` environment variable with your OpenAI
API key for the server to work.  Without a valid key the API will
return a server error.

Example usage:

    export OPENAI_API_KEY="sk-..."
    pip install flask flask-cors openai
    python server.py

Then send a POST request to http://localhost:5000/chat with a JSON
body like {"question": "What are his major achievements?"} and the
model will respond based on the knowledge base.
"""

import os
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS  # type: ignore
except ImportError:
    # If flask_cors is not available, CORS support will be disabled.
    CORS = None  # type: ignore

try:
    # Use the OpenAI v1 client for chat completions.  This is required
    # for API versions >=1.0.0.  If the package is unavailable it will
    # raise ImportError.
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "The openai package is required. Install it with 'pip install openai'."
    ) from exc


def _get_api_key() -> str:
    """Read the OpenAI API key from the environment or raise a runtime error."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is missing. Set this variable "
            "in the Render service environment or your local shell."
        )
    return api_key


app = Flask(__name__)

# Enable CORS for the `/chat` endpoint so that the front‑end hosted on
# GitHub Pages can make requests across domains.  Adjust the origins
# list if you deploy the front‑end to a custom domain.
if CORS:
    CORS(app, resources={r"/chat": {"origins": [
        "https://dadimitris.github.io",
        "https://dadimitris.github.io/ask-and-learn-about-me"
    ]}})


# Build a concise knowledge base from Dimitris Dakos’s CV and LinkedIn
# information.  This should not include sensitive personal data or
# speculative content.  Keep it focused on verifiable roles,
# achievements, education, certifications and interests.
KNOWLEDGE = """
Dimitris Dakos is a seasoned supply‑chain and logistics executive with more than two decades of leadership experience across manufacturing, retail and service environments.

Current role (2024–present): Director of Warehousing at OB Streem — manages multiple sites totalling 239,000 m² of property and 90,300 m² of buildings; leads around 130 people; oversees budgets of €9–12 million; opens new facilities; procures equipment; coaches his team; delivered year‑on‑year cost–service improvements of roughly 19 % while improving margins.

Previous role (2008–2024): Head of Distribution Center at AB Vassilopoulos (Ahold Delhaize) — ran a 33,000 m² distribution centre on a 112,000 m² site; served over 130 stores (up from 34); managed around 180 staff and 10,000+ SKUs with budgets of €7–10 million; led projects to expand capacity, automate workflows and introduce an export hub.  Achievements include +5 % picking productivity, −7 % errors and shrinkage, −10 % yard wait time, +15 % reception capacity, workflow automation saving 10 % work hours, €60,000 annual savings on frozen products processes, automation saving two full‑time equivalents and growing the centre’s capability from 34 to 133 stores.

Earlier career (2002–2008): Head of Logistics at Philkeram Johnson (Norcros Group) — founded the logistics department; coordinated purchasing, demand planning, production scheduling and warehousing; managed a €6 million budget and about 30 people; implemented a stock‑management plan that generated €2.2 million in cash flow over three years; designed an order‑management system that improved service levels by 9 %, lowered inventory carrying costs by 17 % and halved the wait time for backorder information; rolled out a radio‑frequency warehouse management system and supported a fivefold increase in SKUs.

Education: MSc in Logistics Management (University of Sheffield’s CITY College, 2003–2005, distinction); MSc in Marketing Management (Aston University, 1997–1998); BA in Business Studies (University of Sheffield, 1994–1997).  Certifications include PMP (Project Management Professional), MITx Manufacturing Systems I & II, supply‑chain analytics foundations, agile methodologies and AI/ML fundamentals.

Professional affiliations: Board member of the Hellenic Society of Logistics; mentor and guest lecturer on logistics topics.

Interests: Supply‑chain strategy, warehousing and distribution, project management, S&OP, demand planning, procurement, agile and lean methodologies, continuous improvement, leadership development, emerging technologies (AI, machine learning, data science, blockchain/NFTs), education and lifelong learning, diversity and inclusion, sustainability and climate action.  He advocates for democratising AI through no‑code tools and frequently shares insights on innovation, education and leadership.
"""


SYSTEM_PROMPT = (
    "You are a helpful assistant who answers questions about Dimitris Dakos based strictly on the provided KNOWLEDGE. "
    "Do not invent facts or speculate. If the knowledge base does not contain the answer, reply that you do not have that information."
)


def build_messages(question: str) -> list:
    """Construct the conversation messages for the OpenAI API."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"KNOWLEDGE:\n{KNOWLEDGE}\n\nQuestion: {question}"
        },
    ]


@app.route("/chat", methods=["POST"])
def chat() -> tuple:
    """Chat endpoint: accepts a JSON question and returns a model‑generated answer."""
    try:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Please provide a question."}), 400

        # Create OpenAI client per request to avoid global state issues
        client = OpenAI(api_key=_get_api_key())
        messages = build_messages(question)
        # Use a lightweight model for cost efficiency; adjust model as needed
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as exc:
        # On any error, return the exception message to aid debugging
        return jsonify({"error": f"Server error: {type(exc).__name__}: {exc}"}), 500


@app.route("/", methods=["GET"])
def index() -> tuple:
    """Root endpoint that can be used as a simple health check."""
    return "OK", 200


if __name__ == "__main__":  # pragma: no cover
    # Run the development server when executed directly.  In production
    # (Render) this block is ignored because the platform runs the
    # application via the specified start command.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)