python -m rag_system

python -m rag_system api start

curl -X POST http://localhost:8000/api/ingest -F "file=@document.pdf"

curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "return_sources": true}'


python -m rag_system ingest document.pdf
