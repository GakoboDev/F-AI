# app.py
from flask import Flask
from db import init_db_connection_pool
from routes.user_routes import user_routes

app = Flask(__name__)

# Initialize the database connection pool
init_db_connection_pool()

# Register the user routes
app.register_blueprint(user_routes)

if __name__ == '__main__':
    app.run(debug=True)
