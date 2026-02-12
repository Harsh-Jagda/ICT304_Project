import csv
from flask import session
from config import Config


# ---------------- LOAD USERS ----------------
def load_users():

    users = {}

    with open(Config.USERS_FILE, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            users[row["username"]] = {
                "password": row["password"],
                "role": row["role"]
            }

    return users


# ---------------- LOGIN CHECK ----------------
def check_login(username, password):

    users = load_users()

    if username not in users:
        return None

    if users[username]["password"] == password:
        return users[username]["role"]

    return None


# ---------------- SESSION ----------------
def login_user(username, role):

    session["logged_in"] = True
    session["username"] = username
    session["role"] = role


def logout_user():

    session.clear()


# ---------------- ACCESS CONTROL ----------------
def role_required(allowed_roles):

    if "logged_in" not in session:
        return False

    if session.get("role") not in allowed_roles:
        return False

    return True


# ---------------- HELPERS ----------------
def get_role():
    return session.get("role")


def get_username():
    return session.get("username")

