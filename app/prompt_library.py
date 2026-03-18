PROMPT_LIBRARY = {
    "Executive Briefs": {
        "description": "High-level summaries and decision-ready outputs for leadership and stakeholders.",
        "prompts": [
            "Give me a five-bullet executive summary across all uploaded files.",
            "What are the top three business decisions these documents support?",
            "Summarize the biggest opportunities, risks, and next steps for leadership.",
            "Create a board-ready brief with key findings, implications, and action items.",
            "What changed across these documents that an executive should care about first?",
            "Condense all uploaded files into a one-minute leadership update.",
        ],
    },
    "Cross-Document Comparison": {
        "description": "Compare multiple reports, decks, sheets, and datasets side by side.",
        "prompts": [
            "Compare the key themes across all uploaded documents.",
            "What are the biggest differences in recommendations between these files?",
            "Create a comparison matrix of goals, risks, and assumptions across the documents.",
            "Which findings are consistent across files and which ones conflict?",
            "Compare the evidence quality and confidence level across the uploaded sources.",
            "Where do the documents disagree on priorities, timelines, or budgets?",
        ],
    },
    "Presentation Review": {
        "description": "Analyze PowerPoint decks for narratives, gaps, slide quality, and messaging.",
        "prompts": [
            "Summarize the storyline of the uploaded presentation in plain English.",
            "What is the main message each slide is trying to communicate?",
            "Which slides feel weak, repetitive, or unsupported by evidence?",
            "Turn this presentation into a concise speaker brief.",
            "What objections or questions would an audience likely ask after seeing this deck?",
            "Extract the strongest proof points and action items from the uploaded slides.",
        ],
    },
    "Spreadsheet And CSV Analysis": {
        "description": "Explore metrics, rows, anomalies, and business signals in structured data files.",
        "prompts": [
            "What are the most important metrics in the uploaded spreadsheet or CSV files?",
            "Highlight trends, outliers, and anomalies across the tabular data.",
            "Which rows or records look risky, incomplete, or unusual?",
            "Summarize the dataset by segment, category, or time period if possible.",
            "What decisions could a business analyst make from these sheets and CSVs?",
            "Turn the spreadsheet findings into a short narrative for non-technical stakeholders.",
        ],
    },
    "Sales And Marketing": {
        "description": "Extract messaging, objections, customer value, and competitive angles.",
        "prompts": [
            "What are the strongest customer value propositions in these files?",
            "List the top objections, concerns, or friction points mentioned.",
            "Summarize the target audience and positioning implied by the documents.",
            "What proof points or metrics would strengthen a sales pitch based on this material?",
            "Extract customer pain points, desired outcomes, and differentiators.",
            "Create a concise messaging framework from the uploaded content.",
        ],
    },
    "Customer Research And UX": {
        "description": "Synthesize user interviews, survey exports, notes, and research decks.",
        "prompts": [
            "What major user pain points show up across these research documents?",
            "Group the findings into themes, patterns, and unmet needs.",
            "What user quotes, survey patterns, or observations matter most?",
            "Identify the most urgent UX issues and likely design opportunities.",
            "Which user segments seem to have different needs or expectations?",
            "Turn these research files into prioritized product insights.",
        ],
    },
    "Finance And Forecasting": {
        "description": "Review revenue sheets, budgets, forecasts, and finance decks.",
        "prompts": [
            "Summarize the financial health signals in the uploaded files.",
            "What revenue, cost, margin, or growth patterns stand out most?",
            "Identify budget risks, forecast assumptions, and likely financial pressure points.",
            "Which figures deserve validation because they look inconsistent or risky?",
            "Compare actuals versus plans if that information appears in the files.",
            "What would a finance leader want to challenge or validate here?",
        ],
    },
    "Operations And Delivery": {
        "description": "Analyze execution plans, trackers, operational dashboards, and status files.",
        "prompts": [
            "What are the biggest operational blockers or bottlenecks in these files?",
            "Summarize current delivery status, risks, and dependencies.",
            "Which workstreams look off-track, delayed, or under-resourced?",
            "Create an action-oriented status summary for operations leadership.",
            "What process gaps or coordination issues can you infer from the documents?",
            "What should an operations manager do next based on this evidence?",
        ],
    },
    "Risk And Compliance": {
        "description": "Surface issues, gaps, controls, unresolved concerns, and mitigation ideas.",
        "prompts": [
            "List the top risks mentioned across all uploaded files.",
            "Which compliance, audit, or governance concerns should be escalated?",
            "Summarize the evidence for each major risk and its likely impact.",
            "Where do the documents reveal ambiguity, weak controls, or missing decisions?",
            "Create a risk register style summary from the uploaded files.",
            "What mitigation steps are supported by the evidence in these documents?",
        ],
    },
    "Product And Roadmap": {
        "description": "Extract requirements, themes, opportunities, roadmap implications, and tradeoffs.",
        "prompts": [
            "What product requirements or feature ideas are implied by these files?",
            "What are the clearest signals for roadmap priority changes?",
            "Summarize the biggest product opportunities, risks, and open questions.",
            "Which user problems appear repeatedly across the uploaded documents?",
            "Turn this content into a lightweight product requirements brief.",
            "What tradeoffs should a product manager consider based on this evidence?",
        ],
    },
    "Strategy And Planning": {
        "description": "Support planning, market reviews, strategic synthesis, and decision framing.",
        "prompts": [
            "What strategic themes or directional bets emerge from these files?",
            "What assumptions should be tested before acting on these recommendations?",
            "Create a strategy memo based on the uploaded content.",
            "Where are the strongest growth opportunities and the biggest strategic risks?",
            "What decisions would benefit from more data or validation?",
            "Summarize this material into options, tradeoffs, and recommended next steps.",
        ],
    },
    "Knowledge Extraction": {
        "description": "Use the app as a general-purpose QA and extraction workspace.",
        "prompts": [
            "Extract every important date, deadline, and milestone from the uploaded files.",
            "What names, teams, owners, or stakeholders are mentioned most often?",
            "List the metrics, KPIs, and targets referenced across the documents.",
            "What facts, evidence, or claims are repeated across multiple files?",
            "What are the most important unanswered questions after reviewing these files?",
            "Create a concise FAQ from the uploaded documents.",
        ],
    },
}


def get_prompt_categories():
    return list(PROMPT_LIBRARY.keys())
