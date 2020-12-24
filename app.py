from flask import Flask, request, redirect, url_for, current_app
from werkzeug.utils import secure_filename

from terms_extraction import *

UPLOAD_FOLDER = 'save'
ALLOWED_EXTENSIONS = set(['zip'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Terms extraction</title>
    <h1>Terms extraction</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=extraction>
    </form>
    '''


from flask import send_from_directory


@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def uploaded_file(filename):
    fantasy_zip = zipfile.ZipFile(f'save/{filename}')

    fantasy_zip.extractall('save/Samples for term extractor')

    fantasy_zip.close()
    extract = TermsExtraction(directory='save/Samples for term extractor', nlp=spacy.load("en_core_web_sm"))
    extract.conveyor(remove_all_except_letter_dot=remove_all_except_letter_dot_eng,
                     remove_stop_words=remove_stop_words_eng)
    extract.cleaning()

    df = extract.tf_idf(corpus=extract.corpus_dict, res=extract.get_clean_terms())
    df = df.sort_values('terms_score', ascending=False)
    df.to_csv('save/result.csv', index=False)

    return send_from_directory(app.config['UPLOAD_FOLDER'], 'result.csv', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)