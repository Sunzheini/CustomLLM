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


def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """Search for papers on arXiv related to the given topic and save their info to a JSON file."""
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
        paper_id = paper.get_short_id()
        paper_ids.append(paper_id)

        # Only add if not already present to avoid overwriting
        if paper_id not in papers_info:
            paper_info = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "url": paper.pdf_url,
                "published": str(paper.published.date()),
            }
            papers_info[paper_id] = paper_info

    # Save updated papers info to JSON file
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")
    return paper_ids


def extract_info(paper_id: str) -> str:
    """Extract and return the saved info for a given paper ID."""
    # Check if PAPER_DIR exists
    if not os.path.exists(PAPER_DIR):
        return f"Directory {PAPER_DIR} does not exist"

    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)

        # Check if it's a directory
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")

            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as json_file:
                        papers_info = json.load(json_file)

                    if paper_id in papers_info:
                        return json.dumps(papers_info[paper_id], indent=2)

                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There is no saved info related to paper id: {paper_id}"


# ---------------------------------------------------------------------------
# Define tools
# ---------------------------------------------------------------------------
tools = [
    {
        "name": "search_papers",
        "description": "Search for research papers on arXiv related to a given topic. "
                       "Input should be a topic string and an optional max_results integer. "
                       "Returns a list of paper IDs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to search for."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "extract_info",
        "description": "Extract and return the saved information for a given paper ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The arXiv paper ID to extract information for."
                }
            },
            "required": ["paper_id"]
        }
    }
]

# ToDO: 3.48



if __name__ == "__main__":
    all_paper_ids = search_papers("computer vision", max_results=2)
    all_paper_ids.append("414567890")  # non-existing paper id for testing

    for pid in all_paper_ids:
        info = extract_info(pid)
        print(f"Info for paper ID {pid}:\n{info}\n")
