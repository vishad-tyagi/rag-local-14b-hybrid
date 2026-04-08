import os
from pathlib import Path
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import your existing local MLX model wrapper
from hf_models import LangChainMLXLLM

# Configuration - Match your Neo4j Desktop settings
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password@4141"  # Change this to the password you set in Neo4j Desktop
DATA_PATH = Path("data/sample.txt")

def main():
    # 1. Initialize Connection to Neo4j
    print("Connecting to Neo4j...")
    graph = Neo4jGraph(
        url=NEO4J_URL, 
        username=NEO4J_USER, 
        password=NEO4J_PASSWORD
    )

    # 2. Load and Split Document (Reusing your sample.txt)
    print(f"Loading {DATA_PATH}...")
    loader = TextLoader(str(DATA_PATH))
    raw_documents = loader.load()

    # We use slightly smaller chunks for Graph Extraction to help the 
    # 7B model focus on specific entity relationships per pass.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks for graph processing.")

    # 3. Initialize your 7B model for Extraction
    # The 'Coder' variant is excellent at following extraction schemas.
    llm = LangChainMLXLLM()
    
    # Define the 'Transformer' that converts text -> nodes/edges
    # We leave allowed_nodes/relationships empty so the model discovers them.
    # Define the 'Transformer' with a strict schema to guide the 7B model
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[
            "Technology", "Algorithm", "Concept", "Model", 
            "Library", "Optimization", "Metric", "IndexType"
        ],
        allowed_relationships=[
            "USES", "IMPLEMENTS", "OPTIMIZES", 
            "PART_OF", "RELATED_TO", "EVALUATES"
        ]
    );l/ 

    # 4. Extract and Store the Graph
    print("Extracting entities and relationships (this may take a few minutes)...")
    try:
        # This step uses your MacBook GPU (via MLX) to 'read' the text
        graph_documents = transformer.convert_to_graph_documents(documents)
        
        print(f"Adding {len(graph_documents)} graph documents to Neo4j...")
        graph.add_graph_documents(
            graph_documents, 
            baseEntityLabel=True, 
            include_source=True
        )
        print("Success! Your GraphDB is now populated.")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    main()