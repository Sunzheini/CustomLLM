import json
import os
from typing import List

import arxiv            # open source repository for many published papers
import anthropic
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Streamlit example (use the streamlit_runner in the configurations to run)
# ---------------------------------------------------------------------------
from streamlit_example.example_backend import ExampleBackend
from streamlit_example.example_streamlit_frontend import StreamlitFrontend


# Option 1. run with the streamlit runner configuration
# backend = ExampleBackend()
# frontend = StreamlitFrontend(backend)  # runs in a loop

# Option 2. run a service for AegisAI: use the uvicorn.exe configuration to run


# ToDo: chatbot example -> transfer to a test
PAPER_DIR = "papers"


def search_papers(topic: str, max_results: int=5) -> List[str]:
    # Use arxiv library to search for papers
    client = arxiv.Client()

    # Search for the most relevant articles matching the queries topic
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = client.results(search)

    # Create a directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "url": paper.pdf_url,
            "published": str(paper.published.date()),
        }
        papers_info[paper.get_short_id()] = paper_info

    # Save updated papers info to JSON file
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")
    return paper_ids



















