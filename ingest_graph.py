import os
import re
from pathlib import Path

from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hf_models import LangChainMLXLLM


NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password@4141")
DATA_PATH = Path("data/graph_sample.txt")
RESET_GRAPH_BEFORE_INGEST = os.getenv("RESET_GRAPH_BEFORE_INGEST", "false").lower() == "true"
RUN_LLM_GRAPH_EXTRACTION = os.getenv("RUN_LLM_GRAPH_EXTRACTION", "true").lower() == "true"

ALLOWED_NODE_LABELS = {
    "Technology",
    "Algorithm",
    "Concept",
    "Model",
    "Library",
    "Optimization",
    "Metric",
    "IndexType",
    "Assistant",
    "Pipeline",
}
ALLOWED_RELATIONSHIPS = {
    "USES",
    "IMPLEMENTS",
    "OPTIMIZES",
    "PART_OF",
    "RELATED_TO",
    "EVALUATES",
}


def _clean_entity_id(text: str) -> str:
    cleaned = text.strip().strip(".")
    cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_structured_facts(text: str):
    labels = {}
    relationships = []

    label_pattern = re.compile(
        r"^(?P<entity>.+?)\s+is\s+(?:a|an)\s+(?P<label>[A-Za-z]+)\.$",
        flags=re.IGNORECASE,
    )
    relationship_pattern = re.compile(
        r"^(?P<source>.+?)\s+(?P<relationship>USES|IMPLEMENTS|OPTIMIZES|PART_OF|RELATED_TO|EVALUATES)\s+(?P<target>.+?)\.$"
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("title:"):
            continue

        label_match = label_pattern.match(line)
        if label_match:
            label = label_match.group("label")
            if label in ALLOWED_NODE_LABELS:
                labels[_clean_entity_id(label_match.group("entity"))] = label
            continue

        relationship_match = relationship_pattern.match(line)
        if relationship_match:
            relationship = relationship_match.group("relationship")
            if relationship in ALLOWED_RELATIONSHIPS:
                relationships.append(
                    {
                        "source": _clean_entity_id(relationship_match.group("source")),
                        "relationship": relationship,
                        "target": _clean_entity_id(relationship_match.group("target")),
                    }
                )

    return labels, relationships


def _merge_node(graph: Neo4jGraph, entity_id: str, label: str | None = None) -> None:
    if label and label not in ALLOWED_NODE_LABELS:
        raise ValueError(f"Unsupported node label: {label}")

    query = "MERGE (n:Entity {id: $id})"
    if label:
        query += f" SET n:`{label}`"

    graph.query(query, params={"id": entity_id})


def _merge_relationship(graph: Neo4jGraph, source: str, relationship: str, target: str) -> None:
    if relationship not in ALLOWED_RELATIONSHIPS:
        raise ValueError(f"Unsupported relationship type: {relationship}")

    graph.query(
        f"""
        MERGE (source:Entity {{id: $source}})
        MERGE (target:Entity {{id: $target}})
        MERGE (source)-[:`{relationship}`]->(target)
        """,
        params={"source": source, "target": target},
    )


def add_structured_graph_facts(graph: Neo4jGraph, text: str) -> None:
    labels, relationships = _extract_structured_facts(text)

    for entity_id, label in labels.items():
        _merge_node(graph, entity_id, label)

    for relationship in relationships:
        _merge_node(graph, relationship["source"], labels.get(relationship["source"]))
        _merge_node(graph, relationship["target"], labels.get(relationship["target"]))
        _merge_relationship(
            graph,
            relationship["source"],
            relationship["relationship"],
            relationship["target"],
        )

    print(
        f"Added deterministic graph facts: {len(labels)} labeled nodes, "
        f"{len(relationships)} relationships."
    )


def main():
    print("Connecting to Neo4j...")
    graph = Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )

    print(f"Loading {DATA_PATH}...")
    loader = TextLoader(str(DATA_PATH), encoding="utf-8")
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks for graph processing.")

    full_text = "\n\n".join(doc.page_content for doc in raw_documents)

    try:
        if RESET_GRAPH_BEFORE_INGEST:
            print("RESET_GRAPH_BEFORE_INGEST=true, clearing existing Neo4j graph first...")
            graph.query("MATCH (n) DETACH DELETE n")

        add_structured_graph_facts(graph, full_text)

        if RUN_LLM_GRAPH_EXTRACTION:
            llm = LangChainMLXLLM()

            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=sorted(ALLOWED_NODE_LABELS),
                allowed_relationships=sorted(ALLOWED_RELATIONSHIPS),
            )

            print("Extracting additional graph facts with the LLM (this may take a few minutes)...")
            graph_documents = transformer.convert_to_graph_documents(documents)
            print(f"Adding {len(graph_documents)} graph documents to Neo4j...")
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True,
            )

        print("Success! Your GraphDB is now populated.")
        if not RESET_GRAPH_BEFORE_INGEST:
            print("Tip: set RESET_GRAPH_BEFORE_INGEST=true before rerunning if you want a clean graph.")
    except Exception as e:
        print(f"Error during ingestion: {e}")


if __name__ == "__main__":
    main()
