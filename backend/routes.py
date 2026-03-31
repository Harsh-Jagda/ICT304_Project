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

    uploads_list = []
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    if os.path.exists(upload_folder):
        uploads_list = os.listdir(upload_folder)

    active_csv = session.get("active_csv", "None")
    
    # ✅ Extract just the filename for display
    if active_csv and active_csv != "None":
        active_name = os.path.basename(active_csv)
    else:
        active_name = "None"

    return render_template(
        "dashboard.html",
        username=get_username(),
        role=get_role(),
        uploads=uploads_list,
        active_name=active_name
    )



@routes.route("/forecast")
def forecast():
    csv_path = session.get("active_csv")
    
    print(f"🔍 ROUTE DEBUG: active_csv = {csv_path}")

    if not csv_path or not os.path.exists(csv_path):
        return f"Error: No valid dataset. Session has: {csv_path}, Exists: {os.path.exists(csv_path) if csv_path else False}"

    from services.forecasting_service import run_forecast
    results, err = run_forecast(csv_path)

    if err:
        return f"<h2>Forecast Error</h2><pre>{err}</pre><br><a href='/dashboard'>Back to Dashboard</a>"

    print(f"✅ ROUTE DEBUG: Forecast successful: {results}")
    return render_template("forecast.html", data=results)




@routes.route("/risk")
def risk():
    csv_path = session.get("active_csv")

    if not csv_path or not os.path.exists(csv_path):
        flash("No valid dataset selected. Please select a CSV or use demo data.")
        return redirect(url_for("routes.dashboard"))

    from services.risk_service import compute_risk

    # ✅ Calculate risk for all three categories
    categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    results = []

    for cat in categories:
        result, err = compute_risk(csv_path, category=cat)
        if not err:
            results.append(result)

    if not results:
        return f"<h2>Risk Analysis Error</h2><pre>Could not calculate risk for any category</pre><br><a href='/dashboard'>Back to Dashboard</a>"

    return render_template("risk.html", data=results)  # Pass list of results


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

    # ✅ Store FULL path instead of just filename
    full_path = os.path.join(current_app.config["UPLOAD_FOLDER"], selected_file)
    session["active_csv"] = full_path

    return redirect(url_for("routes.dashboard"))


@routes.route("/demo")
def demo():
    demo_path = current_app.config["DEMO_DATA_PATH"]
    
    print(f"🔍 DEBUG DEMO: Looking for file at: {demo_path}")
    print(f"🔍 DEBUG DEMO: File exists: {os.path.exists(demo_path)}")
    
    if not os.path.exists(demo_path):
        return f"Demo file not found at: {demo_path}<br><a href='/dashboard'>Back</a>"
    
    session["active_csv"] = demo_path
    return redirect(url_for("routes.dashboard"))



@routes.route("/instructions")
def instructions():
    return render_template("instructions.html")
