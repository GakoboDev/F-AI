# routes/user_routes.py
from flask import Blueprint, request, jsonify
from controllers.user_controller import add_user, view_users, view_user_by_id

user_routes = Blueprint('user_routes', __name__)

# Route to add a user
@user_routes.route('/add_user', methods=['POST'])
def add_user_route():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    return add_user(username, email)

# Route to view users
@user_routes.route('/view_users', methods=['GET'])
def view_users_route():
    return view_users()

# Route to view a specific user by ID
@user_routes.route('/view_user/<int:user_id>', methods=['GET'])
def view_user_route(user_id):
    return view_user_by_id(user_id)