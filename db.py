# db.py
import psycopg2
from psycopg2 import pool

# Global database connection pool
db_pool = None

def init_db_connection_pool():
    global db_pool
    db_pool = psycopg2.pool.SimpleConnectionPool(
        1,  # Minimum number of connections
        10,  # Maximum number of connections
        dbname='pytest1',
        user='gakobo2',
        password='development',
        host='localhost',
        port='5432'
    )

def get_db_connection():
    return db_pool.getconn()

def return_db_connection(conn):
    db_pool.putconn(conn)
