from flask import Flask

# initiate the app
app = Flask(__name__)

app.route('/')
def hello():
    return "<p>Hello, World!</p>"