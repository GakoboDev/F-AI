# controllers/user_controller.py
from flask import jsonify
from models.user import User
from db import get_db_connection, return_db_connection

def add_user(username, email):
    conn = get_db_connection()
    User.add_user(username, email, conn)
    return_db_connection(conn)
    return jsonify({"message": "User added successfully!"}), 201

def view_users():
    conn = get_db_connection()
    users = User.get_all_users(conn)
    return_db_connection(conn)

    # Prepare the data to return
    user_data = [{"id": user[0], "username": user[1], "email": user[2], "created_at": user[3]} for user in users]

    return jsonify({"message": "Users fetched successfully!", "data": user_data}), 200

def view_user_by_id(user_id):
    conn = get_db_connection()
    user = User.get_user_by_id(user_id, conn)
    return_db_connection(conn)

    if user:
        user_data = {
            "id": user[0],
            "username": user[1],
            "email": user[2],
            "created_at": user[3]
        }
        return jsonify({"message": "User fetched successfully!", "data": user_data}), 200
    else:
        return jsonify({"message": "User not found!"}), 404