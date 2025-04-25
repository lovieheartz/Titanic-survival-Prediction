# run.py

# This is the main script that starts the Flask app.

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
