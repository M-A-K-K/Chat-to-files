from flask import request, jsonify
from .file_processing import process_file, allowed_file
from .pinecone_utils import get_index, upsert_embeddings

def handle_file_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('file')

    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No files selected"}), 400

    all_embeddings = []
    for file in files:
        if file and allowed_file(file.filename):
            embeddings = process_file(file)
            all_embeddings.extend(embeddings)
        else:
            return jsonify({"error": f"You are not allowed to upload this type of file: {file.filename}"}), 400

    index = get_index()
    upsert_embeddings(index, all_embeddings)

    response_data = {"embeddings": all_embeddings}
    return jsonify(response_data), 200
