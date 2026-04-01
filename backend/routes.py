from datetime import datetime
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

GLOBAL_STATE = {
    "active_csv": None,
    "active_json": "lead_times.json"
}


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

    # 1. Scan for CSV/Parquet files in Uploads
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    uploads_list = []
    if os.path.exists(upload_folder):
        uploads_list = [f for f in os.listdir(upload_folder) if f.endswith(('.csv', '.parquet'))]

    # 2. Scan for JSON configs in Model and Uploads
    json_list = []
    model_folder = os.path.join(current_app.root_path, "model")
    if os.path.exists(model_folder):
        json_list.extend([f for f in os.listdir(model_folder) if f.endswith('.json')])
    
    if os.path.exists(upload_folder):
        user_jsons = [f for f in os.listdir(upload_folder) if f.endswith('.json')]
        for uj in user_jsons:
            if uj not in json_list:
                json_list.append(uj)

    # 3. Resolve the GLOBAL active state (Not from Session)
    active_path = GLOBAL_STATE["active_csv"]
    
    if active_path and os.path.exists(active_path):
        active_name = os.path.basename(active_path)
    else:
        active_name = "None"

    active_json = GLOBAL_STATE["active_json"]

    # 4. Load Audit Logs (Admin Only)
    audit_logs = []
    if session.get("role") == "admin":
        log_path = os.path.join(current_app.root_path, "data", "system_audit.log")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                audit_logs = f.readlines()[-5:]

    return render_template(
        "dashboard.html",
        role=session.get("role"),
        uploads=uploads_list,
        json_configs=json_list,
        active_name=active_name,
        active_json=active_json,
        audit_logs=audit_logs
    )


@routes.route("/forecast")
def forecast():
    # Analysts and Admins can trigger the forecast, Viewers can only view existing results
    if not role_required(["admin", "analyst", "viewer"]):
        return redirect(url_for("routes.login"))

    csv_path = GLOBAL_STATE["active_csv"]
    if not csv_path or not os.path.exists(csv_path):
        flash("No active dataset. Please contact an Admin.")
        return redirect(url_for("routes.dashboard"))

    from services.forecasting_service import run_forecast
    results, err = run_forecast(csv_path)
    
    if not err and session.get("role") != "viewer":
        log_system_event("ANALYSIS", "Generated ML Forecast")

    return render_template("forecast.html", data=results, role=session.get("role"))




@routes.route("/risk")
def risk():
    if not role_required(["admin", "analyst", "viewer"]):
        return redirect(url_for("routes.login"))

    csv_path = GLOBAL_STATE["active_csv"]
    active_json = GLOBAL_STATE["active_json"]

    if not csv_path or not os.path.exists(csv_path):
        flash("Dataset missing.")
        return redirect(url_for("routes.dashboard"))

    from services.risk_service import compute_risk
    import pandas as pd

    try:
        temp_df = pd.read_parquet(csv_path, columns=["cat_id"]) if csv_path.endswith(".parquet") else pd.read_csv(csv_path, usecols=["cat_id"])
        categories = temp_df["cat_id"].unique().tolist()
    except:
        categories = ["All Items"]

    results = []
    for cat in categories:
        result, err = compute_risk(csv_path, category=cat, json_filename=active_json)
        if not err: results.append(result)

    return render_template("risk.html", data=results, active_config=active_json, role=session.get("role"))


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
    if not role_required(["admin", "analyst"]):
        flash("Unauthorized access.")
        return redirect(url_for("routes.dashboard"))

    selected_file = request.form.get("selected_file")
    
    if selected_file:
        full_path = os.path.join(current_app.config["UPLOAD_FOLDER"], selected_file)
        
        if os.path.exists(full_path):
            # Update the GLOBAL STATE so Viewers can see it
            GLOBAL_STATE["active_csv"] = full_path
            
            log_system_event("DATA_ACTIVATION", f"Global dataset set to: {selected_file}")
            flash(f"Global Dataset '{selected_file}' is now active for all users.")
        else:
            flash("Error: File not found.")
    
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
    if not role_required(["admin", "analyst"]):
        flash("Unauthorized access.")
        return redirect(url_for("routes.dashboard"))

    selected_json = request.form.get("selected_json")
    
    if selected_json:
        # Update the GLOBAL STATE
        GLOBAL_STATE["active_json"] = selected_json
        
        log_system_event("CONFIG_CHANGE", f"Global config set to: {selected_json}")
        flash(f"Global Config '{selected_json}' is now active for all users.")

    return redirect(url_for("routes.dashboard"))




# INTERNAL HELPER: Audit Logging (New Feature for Admin Oversight)
def log_system_event(event_type, details):
    log_path = os.path.join(current_app.root_path, "data", "system_audit.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] USER: {session.get('username')} | EVENT: {event_type} | {details}\n")


