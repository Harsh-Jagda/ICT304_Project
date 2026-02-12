What’s Added So Far:
-------------------
1. Backend setup using Flask:
   - app.py: application factory and runs the server
   - routes.py: all web routes (login, logout, dashboard, upload, forecast placeholder, risk placeholder)
   - auth.py: user authentication helper functions (reads users.csv)
   - config.py: configuration (paths for templates, uploads, users CSV, secret key)
2. Frontend templates in dark mode:
   - login.html: login page for all users
   - dashboard.html: main dashboard with role-based content
   - upload.html: page for CSV file upload
   - forecast.html: placeholder page for forecasting using selected CSV
   - risk.html: placeholder page for risk analysis using selected CSV
3. Role-based login:
   - Admin: can upload/select CSV, access all features
   - Analyst: can access forecasting and risk pages
   - Viewer: can view dashboard only
4. CSV handling:
   - Users stored in `data/users.csv`
   - Uploaded CSVs saved to `uploads/` folder
   - Session stores the active CSV selection for forecasting/risk
5. Git setup:
   - .gitignore excludes venv, uploads, __pycache__, OS files
   - requirements.txt added for Python dependencies

What Needs to Be Done:
----------------------
- Connect AI/ML models for forecasting and risk analysis
- Implement results display and visualization in forecast.html and risk.html
- Add error handling for uploads, missing CSV selection, and role-based restrictions
- Optional: charts/graphs for demo purposes

How to Run:
-----------
1. Clone repo:
   git clone <repo-url>
2. Navigate to project root:
   cd ICT304_Project
3. Create and activate virtual environment:
   python -m venv venv
   source venv/Scripts/activate   (Windows)
   source venv/bin/activate       (Linux/Mac)
4. Install dependencies:
   pip install -r requirements.txt
5. Run Flask app:
   python backend/app.py
6. Open browser:
   http://127.0.0.1:5000
7. Login with credentials from `data/users.csv`
8. Use dashboard to upload CSV, select active CSV, and navigate to Forecast or Risk placeholders

Notes:
------
- The current forecast/risk pages are placeholders and just show selected CSV
- Uploads folder stores CSVs locally; they are not pushed to Git
- Role-based content is working; upload buttons visible only to Admin

