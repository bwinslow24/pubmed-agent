import os
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
API_KEY = os.getenv("PUBMED_KEY", None)

@tool
def search_articles(query: str) -> List[str]:
    """
        Search PubMed for articles matching the given query.

        This tool uses the PubMed E-utilities API to search for articles based on the provided query.
        It returns a list of PubMed IDs (PMIDs) for the top 3 matching articles.

        Args:
            query (str): The search query string. This can include keywords, MeSH terms,
                         author names, or any valid PubMed search syntax.

        Returns:
            List[str]: A list of PMIDs (PubMed IDs) for the top 3 articles matching the query.

        Raises:
            requests.RequestException: If there's an error with the API request.
            KeyError: If the expected data structure is not found in the API response.

        Example:
            >>> search_articles("COVID-19 vaccine efficacy")
            ['34345882', '34596608', '35148837']
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 3,
        "api_key": API_KEY
    }
    response = requests.get(f'{BASE_URL}esearch.fcgi', params=params)
    data = response.json()

    id_list = data["esearchresult"]["idlist"]

    return data["esearchresult"]["idlist"]

@tool
def fetch_summary(pmid: str) -> Dict[str, Any]:
    """
        Fetch the summary of a PubMed article using its PMID.

        This tool retrieves detailed information about a specific article from PubMed
        using the E-utilities API. It returns a dictionary containing various fields
        such as title, abstract, authors, and publication date.

        Args:
            pmid (str): The PubMed ID (PMID) of the article to fetch.

        Returns:
            Dict[str, Any]: A dictionary containing the article's summary information.
            Common keys include:
                - 'title': The article's title
                - 'abstract': The article's abstract (if available)
                - 'authors': List of author names
                - 'pubdate': Publication date
                - 'journal': Journal information

        Raises:
            requests.RequestException: If there's an error with the API request.
            KeyError: If the PMID is not found or the response structure is unexpected.

        Example:
            >>> fetch_summary("34345882")
            {
                'title': 'Effectiveness of Covid-19 Vaccines against the B.1.617.2 (Delta) Variant',
                'abstract': 'Background: The B.1.617.2 (delta) variant...',
                'authors': ['Lopez Bernal J', 'Andrews N', 'Gower C', ...],
                'pubdate': '2021 Aug 12',
                'journal': 'N Engl J Med'
            }
    """
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "json",
        "api_key": API_KEY
    }
    response = requests.get(f"{BASE_URL}esummary.fcgi", params=params)
    data = response.json()
    return data["result"][pmid]


@tool
def fetch_article_abstract(pmids: List[str]) -> str:
    """
    Fetch PubMed article abstracts using the EFetch utility.

    Args:
        pmids (List[str]): A list of PubMed IDs (PMIDs) to fetch.
    Returns:
        str: The fetched PubMed article abstracts.

    Raises:
        requests.RequestException: If there's an error with the API request.
    """
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": 'text',
        "rettype": 'abstract',
        "api_key": API_KEY
    }

    response = requests.get(f"{BASE_URL}efetch.fcgi", params=params)
    response.raise_for_status()

    return response.text


if __name__ == "__main__":
    articles = search_articles("COVID-19 vaccine efficacy")
    print(articles)
    summary = fetch_article_abstract({'pmids': articles})
    print(summary)
    # ['39953414', '39947305', '39946827']
