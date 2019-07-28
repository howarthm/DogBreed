#import json
from werkzeug import secure_filename

from flask import Flask
from flask import render_template, request
from dog_app import dog_classifier

app = Flask(__name__)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    
    # render web page with plotly graphs
    return render_template('master.html')



@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        f = request.files['file']
        temp = secure_filename(f.filename)
        #TODO: clean up old files
        f.save('./static/'+temp)
        #breed = classifier.classify_dog_breed('images/sample_human_2.png')
        classifier = dog_classifier('saved_models/resnet50.json','saved_models/weights.best.Resnet50.hdf5')
        breed = classifier.classify_dog_breed('images/sample_human_2.png')
        return render_template('master.html', image='/static/'+temp, breed=breed)


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()