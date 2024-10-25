# models/user.py
class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email

    @classmethod
    def add_user(cls, username, email, db_connection):
        cursor = db_connection.cursor()
        cursor.execute('INSERT INTO users (username, email) VALUES (%s, %s)', (username, email))
        db_connection.commit()
        cursor.close()

    @classmethod
    def get_all_users(cls, db_connection):
        cursor = db_connection.cursor()
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()  # Fetch all results
        cursor.close()
        return users
    
    @classmethod
    def get_user_by_id(cls, user_id, db_connection):
        cursor = db_connection.cursor()
        cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
        user = cursor.fetchone()  # Fetch a single user
        cursor.close()
        return user
