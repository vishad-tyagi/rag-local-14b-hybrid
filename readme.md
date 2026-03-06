Step to install dependency and run project
cd rag-flask-ollama
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama serve

python ingest.py

python app.py

which python

conda deactivate
unalias python
source .venv/bin/activate
command -v python

rm -rf vectorstore/faiss_index

will require three separate terminal
control+C to stop terminal 

