from flask import Flask, url_for
app = Flask(__name__)

@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/hello')
def api_articles():
    return 'Hello World'

if __name__ == '__main__':
    app.run()
