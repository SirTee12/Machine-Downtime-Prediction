from flask import Flask
from flask import render_template

import pandas as pd

# initiate the app
application = Flask(__name__)

app = application

@app.route('/')
def hello():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug = True)