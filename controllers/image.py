import os
from flask import current_app, jsonify
from werkzeug.utils import secure_filename
from . import label_image


def predict_herb_image(request):
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(os.path.dirname(current_app.config['UPLOAD_FOLDER']), exist_ok=True)  # Create directory if does not exist
    print(current_app.config['UPLOAD_FOLDER'])
    print(file_full_path)
    file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
    classification_output = label_image.classify_herb_image(file_full_path)
    return jsonify({'results': classification_output})
