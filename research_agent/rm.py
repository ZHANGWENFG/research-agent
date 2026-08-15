import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Callable, Union, List

import requests

from .utils import WebPageHelper

logger = logging.getLogger(__name__)


class ArxivRM:
    """Retrieve paper metadata from the public arXiv API."""

    def __init__(
        self,
        k=3,
        endpoint="https://export.arxiv.org/api/query",
        sort_by="relevance",
        sort_order="descending",
        is_valid_source: Callable = None,
    ):
        self.k = k
        self.endpoint = endpoint
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.usage = 0
        self.is_valid_source = is_valid_source or (lambda x: True)

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"ArxivRM": usage}

    def request(self, query: str):
        response = requests.get(
            self.endpoint,
            params={
                "search_query": query,
                "start": 0,
                "max_results": self.k,
                "sortBy": self.sort_by,
                "sortOrder": self.sort_order,
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.text

    @staticmethod
    def _normalize_text(text):
        return " ".join((text or "").split())

    @staticmethod
    def _normalize_query_for_arxiv(query):
        query = " ".join((query or "").split())
        if not query:
            return ""

        normalized = query
        replacements = {
            "无源互调": "passive intermodulation",
            "神经网络": "neural network",
            "抑制": "suppression",
            "射频": "radio frequency",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, f" {target} ")
        normalized = re.sub(r"\s+", " ", normalized).strip()

        lower_query = normalized.lower()
        has_pim = re.search(r"\bpim\b", lower_query) is not None
        passive_context_terms = (
            "passive intermodulation",
            "intermodulation",
            "suppression",
            "mitigation",
            "radio frequency",
            " rf",
            "antenna",
            "microwave",
            "neural network",
        )
        memory_context_terms = (
            "processing-in-memory",
            "processing in memory",
            "dram",
            " ram",
            "memory system",
        )

        if has_pim and any(term in lower_query for term in passive_context_terms):
            normalized = re.sub(
                r"\bpim\b", "passive intermodulation", normalized, flags=re.I
            )
        elif has_pim and any(term in lower_query for term in memory_context_terms):
            return ""

        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _is_result_relevant_to_query(query, result):
        query = (query or "").lower()
        if "passive intermodulation" not in query:
            return True

        haystack = " ".join(
            [
                result.get("title") or "",
                result.get("description") or "",
                " ".join(result.get("snippets") or []),
            ]
        ).lower()
        passive_terms = (
            "passive intermodulation",
            "intermodulation",
            "radio frequency",
            "rf ",
            "antenna",
            "microwave",
        )
        off_topic_terms = (
            "processing-in-memory",
            "processing in memory",
            "dram",
            " ram ",
            "memory system",
            "product information management",
        )
        return any(term in haystack for term in passive_terms) and not any(
            term in haystack for term in off_topic_terms
        )

    def _parse_response(self, response_text: str):
        namespace = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(response_text)
        results = []

        for entry in root.findall("atom:entry", namespace):
            paper_id = self._normalize_text(entry.findtext("atom:id", namespaces=namespace))
            title = self._normalize_text(
                entry.findtext("atom:title", namespaces=namespace)
            )
            abstract = self._normalize_text(
                entry.findtext("atom:summary", namespaces=namespace)
            )
            published = self._normalize_text(
                entry.findtext("atom:published", namespaces=namespace)
            )
            updated = self._normalize_text(
                entry.findtext("atom:updated", namespaces=namespace)
            )
            authors = [
                self._normalize_text(author.findtext("atom:name", namespaces=namespace))
                for author in entry.findall("atom:author", namespace)
            ]
            authors = [author for author in authors if author]

            categories = [
                category.attrib.get("term")
                for category in entry.findall("atom:category", namespace)
                if category.attrib.get("term")
            ]
            primary_category = entry.find("arxiv:primary_category", namespace)
            primary_category_term = (
                primary_category.attrib.get("term")
                if primary_category is not None
                else None
            )

            abs_url = paper_id
            pdf_url = None
            for link in entry.findall("atom:link", namespace):
                href = link.attrib.get("href")
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = href
                if link.attrib.get("rel") == "alternate" and href:
                    abs_url = href

            if not all([abs_url, title, abstract]):
                continue

            results.append(
                {
                    "url": abs_url,
                    "title": title,
                    "description": abstract,
                    "snippets": [abstract],
                    "meta": {
                        "source_type": "arxiv",
                        "authors": authors,
                        "published": published,
                        "updated": updated,
                        "categories": categories,
                        "primary_category": primary_category_term,
                        "pdf_url": pdf_url,
                    },
                }
            )

        return results

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [
            self._normalize_query_for_arxiv(query)
            for query in queries
            if query and query.strip()
        ]
        queries = [query for query in queries if query]
        self.usage += len(queries)

        collected_results = []
        seen_urls = set()
        for query in queries:
            try:
                results = self._parse_response(self.request(query))
            except Exception as e:
                logger.info("Skipping failed arXiv query %r: %s", query, e)
                continue

            for result in results:
                url = result["url"]
                if (
                    url in seen_urls
                    or url in exclude_urls
                    or not self.is_valid_source(url)
                    or not self._is_result_relevant_to_query(query, result)
                ):
                    continue
                collected_results.append(result)
                seen_urls.add(url)

        return collected_results


class LocalPDFRM:
    """Retrieve relevant chunks from a local PDF collection."""

    def __init__(
        self,
        pdf_dir=None,
        documents=None,
        k=3,
        chunk_size=1200,
        chunk_overlap=150,
        is_valid_source: Callable = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.k = k
        self.pdf_dir = pdf_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.usage = 0
        self.is_valid_source = is_valid_source or (lambda x: True)
        self.chunks = []

        source_documents = list(documents or [])
        if pdf_dir:
            source_documents.extend(self._load_pdf_documents(pdf_dir))
        for document in source_documents:
            self._add_document(document)

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"LocalPDFRM": usage}

    @staticmethod
    def _tokenize(text):
        return set(re.findall(r"[\w]+", (text or "").lower()))

    @staticmethod
    def _normalize_text(text):
        return " ".join((text or "").split())

    def _load_pdf_documents(self, pdf_dir):
        try:
            from pypdf import PdfReader
        except ImportError as err:
            raise ImportError("LocalPDFRM requires `pip install pypdf`.") from err

        documents = []
        for root, _, files in os.walk(pdf_dir):
            for filename in files:
                if not filename.lower().endswith(".pdf"):
                    continue
                path = os.path.join(root, filename)
                reader = PdfReader(path)
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
                documents.append(
                    {
                        "title": os.path.splitext(filename)[0],
                        "path": path,
                        "text": "\n".join(pages),
                    }
                )
        return documents

    def _iter_chunks(self, text):
        text = self._normalize_text(text)
        if not text:
            return
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(text):
            chunk = text[start : start + self.chunk_size].strip()
            if chunk:
                yield chunk
            start += step

    def _add_document(self, document):
        text = document.get("text", "")
        title = document.get("title") or os.path.basename(document.get("path", ""))
        path = document.get("path") or document.get("url") or title
        for index, chunk in enumerate(self._iter_chunks(text)):
            self.chunks.append(
                {
                    "url": f"{path}#chunk-{index}",
                    "title": title,
                    "description": f"Local PDF chunk from {title}",
                    "snippets": [chunk],
                    "meta": {
                        "source_type": "local_pdf",
                        "pdf_path": path,
                        "chunk_index": index,
                    },
                    "_tokens": self._tokenize(chunk),
                }
            )

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [query.strip() for query in queries if query and query.strip()]
        self.usage += len(queries)
        if not queries or not self.chunks:
            return []

        scored_results = {}
        for query in queries:
            query_tokens = self._tokenize(query)
            for chunk in self.chunks:
                url = chunk["url"]
                if url in exclude_urls or not self.is_valid_source(url):
                    continue
                score = len(query_tokens & chunk["_tokens"])
                if score <= 0:
                    continue
                if url not in scored_results or score > scored_results[url][0]:
                    scored_results[url] = (score, chunk)

        ranked_chunks = sorted(
            scored_results.values(), key=lambda item: item[0], reverse=True
        )
        results = []
        for _, chunk in ranked_chunks[: self.k]:
            result = {key: value for key, value in chunk.items() if key != "_tokens"}
            results.append(result)
        return results

