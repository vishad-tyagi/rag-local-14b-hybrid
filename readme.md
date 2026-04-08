Step to install dependency and run project

-- In first terminal 
python3.11 -m venv .venv
source .venv/bin/activate 

which python 
 -- ( if python is aliased to global version then follow below)
conda deactivate
unalias python
source .venv/bin/activate
command -v python
-- then again run which python 

pip install -r requirements.txt

python init_db.py
-- to generate SQL DB file

rm -rf vectorstore/faiss_index
-- to delete vector files of sample.txt in data folder

python ingest.py
-- to generate vector index for sample.txt in data folder

MATCH (n) DETACH DELETE n;
-- query to delete everything in Neo4j DB

python ingest_graph.py
-- to ingest data in Neo4j db, remember to start Neo4j instance in Neo4j desktop before run this 

python app.py
-- to run main app

-IN SECOND TERMINAL FOR React Frontend 
cd frontend
npm install vite @vitejs/plugin-react --save-dev
npm install react react-dom

-- for running main UI
npm run dev


-- control+C to stop terminal 

