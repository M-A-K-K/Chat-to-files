from flask import Flask
from utils.handlers import handle_file_upload
import os
from dotenv import load_dotenv

load_dotenv()  

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    return handle_file_upload()

if __name__ == '__main__':
    if not os.path.exists(os.getenv('UPLOAD_FOLDER', 'uploads')):
        os.makedirs(os.getenv('UPLOAD_FOLDER', 'uploads'))
    
    app.run(debug=True)
