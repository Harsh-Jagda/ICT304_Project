from flask import Blueprint, render_template, request, redirect, url_for, render_template_string, flash, current_app, session
import os
import json
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



@routes.route("/dashboard")
def dashboard():
    if not role_required(["admin", "analyst", "viewer"]):
        return redirect(url_for("routes.login"))

    # 1. Get CSVs from the uploads folder
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    uploads_list = []
    if os.path.exists(upload_folder):
        uploads_list = [f for f in os.listdir(upload_folder) if f.endswith(('.csv', '.parquet'))]

    # 2. Get JSONs from BOTH folders (The Special Exception)
    json_list = []
    
    # Path A: The permanent model folder (Demo data)
    model_folder = os.path.join(current_app.root_path, "model")
    if os.path.exists(model_folder):
        json_list.extend([f for f in os.listdir(model_folder) if f.endswith('.json')])

    # Path B: The user uploads folder
    if os.path.exists(upload_folder):
        user_jsons = [f for f in os.listdir(upload_folder) if f.endswith('.json')]
        # Only add if not already in the list to avoid duplicates
        for uj in user_jsons:
            if uj not in json_list:
                json_list.append(uj)

    # 3. Handle Active Selection display
    active_csv = session.get("active_csv", "None")
    active_name = os.path.basename(active_csv) if active_csv != "None" else "None"
    active_json = session.get("active_json", "lead_times.json")

    return render_template(
        "dashboard.html",
        role=get_role(),
        uploads=uploads_list,
        json_configs=json_list,
        active_name=active_name,
        active_json=active_json
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
    # 1. Get the active CSV from session
    csv_path = session.get("active_csv")
    if not csv_path or not os.path.exists(csv_path):
        flash("No valid dataset selected. Please select one on the dashboard.")
        return redirect(url_for("routes.dashboard"))

    # 2. Get the active JSON config from session (default to lead_times.json)
    active_json = session.get("active_json", "lead_times.json")

    from services.risk_service import compute_risk
    import pandas as pd

    # 3. Dynamically find categories in the current file
    try:
        if csv_path.endswith(".parquet"):
            temp_df = pd.read_parquet(csv_path, columns=["cat_id"])
        else:
            # Read just the cat_id column to save memory
            temp_df = pd.read_csv(csv_path, usecols=["cat_id"])
        
        categories = temp_df["cat_id"].unique().tolist()
    except Exception:
        categories = ["All Items"]

    results = []

    # 4. Run analysis for each category using the SELECTED JSON
    for cat in categories:
        result, err = compute_risk(
            csv_path, 
            category=cat, 
            json_filename=active_json
        )
        if not err:
            results.append(result)

    if not results:
        flash("Could not generate risk analysis results.")
        return redirect(url_for("routes.dashboard"))

    return render_template("risk.html", data=results, active_config=active_json)



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





@routes.route("/update_lead_times", methods=["GET", "POST"])
def update_lead_times():
    if not role_required(["admin"]):
        return "Admins only."

    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file part")
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        if file and file.filename.endswith('.json'):
            # Switch destination to the dynamic UPLOAD_FOLDER
            upload_dir = current_app.config["UPLOAD_FOLDER"]
            
            # Ensure the uploads directory exists
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
                
            save_path = os.path.join(upload_dir, file.filename)
            file.save(save_path)
            
            flash(f"Config {file.filename} uploaded to uploads folder!")
            return redirect(url_for("routes.dashboard"))
        else:
            flash("Please upload a valid .json file")
            
    return render_template("update_lead.html")





@routes.route("/select_json", methods=["POST"])
def select_json():
    if not role_required(["admin"]):
        return "Admins only."

    selected_json = request.form.get("selected_json")
    if selected_json:
        session["active_json"] = selected_json

    return redirect(url_for("routes.dashboard"))
