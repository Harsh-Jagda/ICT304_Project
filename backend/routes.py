from flask import Blueprint, render_template, request, redirect, url_for, render_template_string, flash, current_app, session
import os
from auth import (
    check_login,
    login_user,
    logout_user,
    role_required,
    get_role,
    get_username
)


routes = Blueprint("routes", __name__)


# ---------------- LOGIN PAGE ----------------
@routes.route("/", methods=["GET", "POST"])
def login():

    error = None

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        role = check_login(username, password)

        if role:
            login_user(username, role)
            return redirect(url_for("routes.dashboard"))

        error = "Invalid username or password"

    return render_template("login.html", error=error)


# ---------------- DASHBOARD ----------------
@routes.route("/dashboard")
def dashboard():
    if not role_required(["admin", "analyst", "viewer"]):
        return redirect(url_for("routes.login"))

    # Get list of uploaded files
    uploads_list = []
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    if os.path.exists(upload_folder):
        uploads_list = os.listdir(upload_folder)

    return render_template(
        "dashboard.html",
        username=get_username(),
        role=get_role(),
        uploads=uploads_list
    )


# ---------------- FORECAST ----------------
@routes.route("/forecast")
def forecast():
    csv_file = session.get("active_csv")
    if not csv_file:
        return redirect(url_for("routes.dashboard"))  # No CSV selected yet

    csv_path = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_file)

    # TODO: Load CSV and do forecasting
    return render_template("forecast.html", csv_file=csv_file)
# ---------------- RISK ----------------
@routes.route("/risk")
def risk():
    csv_file = session.get("active_csv")
    if not csv_file:
        return redirect(url_for("routes.dashboard"))  # No CSV selected yet

    csv_path = os.path.join(current_app.config["UPLOAD_FOLDER"], csv_file)

    # TODO: Load CSV and do risk analysis
    return render_template("risk.html", csv_file=csv_file)
# ---------------- UPLOAD ----------------
@routes.route("/upload", methods=["GET", "POST"])
def upload():
    if not role_required(["admin"]):
        return "Admins only."

    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        filename = file.filename
        save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # Redirect to dashboard so admin can see uploaded files
        return redirect(url_for("routes.dashboard"))

    return render_template("upload.html")

# ---------------- LOGOUT ----------------
@routes.route("/logout")
def logout():

    logout_user()
    return redirect(url_for("routes.login"))

# ---------------- SELECT_CSV -------------
@routes.route("/select_csv", methods=["POST"])
def select_csv():
    if not role_required(["admin"]):
        return "Admins only."

    selected_file = request.form.get("selected_file")
    if not selected_file:
        return "No file selected"

    # Store the selected CSV in session so forecasting/risk can use it
    session["active_csv"] = selected_file

    return redirect(url_for("routes.dashboard"))

