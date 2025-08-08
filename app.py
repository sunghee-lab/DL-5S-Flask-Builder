from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
from datetime import datetime, timezone

from digital_library_components import Stream, Structure, Space, Scenario, Society, DigitalLibrary
from services import IndexingService, RetrievalService

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, 'static', 'generated')
os.makedirs(GENERATED_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev')

DOMAINS = {
    'Streams': ['text', 'image', 'audio'],
    'Structures': ['graph', 'tree', 'relational'],
    'Spaces': ['vector', 'semantic', '2D'],
    'Scenarios': ['search', 'recommendation', 'archiving'],
    'Societies': ['individual', 'group', 'institution']
}

BUILT_LIBS = {}

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        selections = {k: request.form.get(k) for k in DOMAINS.keys()}
        stream_kind = selections['Streams']
        if stream_kind == 'text':
            sample_docs = [
                'Digital libraries enable search and preservation of scholarly works.',
                'Image retrieval uses descriptors and embeddings.',
                'Graphs and structures model relationships between entities.',
                'This document discusses vector spaces and cosine similarity.'
            ]
            stream = Stream(kind='text', description='sample text stream', content=sample_docs)
        elif stream_kind == 'image':
            sample_docs = [
                {'id':'i1','tags':['graph','diagram'],'feature': None},
                {'id':'i2','tags':['photo','portrait'],'feature': None}
            ]
            stream = Stream(kind='image', description='sample image stream', content=sample_docs)
        else:
            stream = Stream(kind=stream_kind, description='sample stream', content=[])

        structure = Structure(kind=selections['Structures'])
        space = Space(name=selections['Spaces'])
        scenario = Scenario(name=selections['Scenarios'])
        society = Society()

        dl = DigitalLibrary(stream=stream, structure=structure, space=space, scenario=scenario, society=society)

        indexer = IndexingService(stream, structure, space, scenario, society)
        index_result = indexer.perform()

        retriever = RetrievalService(stream, structure, space, scenario, society,
                                     index=index_result['inverted_index'],
                                     documents=index_result['documents'],
                                     doc_vectors=index_result['doc_vectors'],
                                     vectorizer=index_result.get('vectorizer'))

        key = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        BUILT_LIBS[key] = {
            'dl': dl,
            'indexer': indexer,
            'retriever': retriever,
            'selections': selections
        }

        # Save scaffold
        safe_ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f"dl_{selections['Streams']}_{selections['Structures']}_{safe_ts}.py"
        filepath = os.path.join(GENERATED_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Generated scaffold for {selections}\nprint('This is a scaffold file')\n")

        return render_template('index.html', domains=DOMAINS, built_key=key, selections=selections, results=None)

    return render_template('index.html', domains=DOMAINS, built_key=None, selections=None, results=None)

@app.route('/search/<key>', methods=['POST'])
def search(key):
    entry = BUILT_LIBS.get(key)
    if entry is None:
        return redirect(url_for('index'))
    retriever: RetrievalService = entry['retriever']
    q = request.form.get('q','')
    results = retriever.perform(q, top_k=10)
    return render_template('index.html', domains=DOMAINS, built_key=key, selections=entry['selections'], results=results, query=q)

@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
