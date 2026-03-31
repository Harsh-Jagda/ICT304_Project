README_for_group_members.txt

What’s Added So Far:
-------------------
1. Backend setup using Flask:
   - app.py: Application factory and server entry point.
   - routes.py: Logic for role-based access, file selection, and dashboard navigation.
   - auth.py: Session-based authentication (reads from data/users.csv).
   - config.py: Centralized path management for uploads, models, and templates.

2. AI & Risk Engine (Integrated):
   - services/forecasting_service.py: Uses a LightGBM model (.pkl) to generate real-time daily, weekly, and monthly demand forecasts.
   - services/risk_service.py: Calculates Safety Stock and Reorder Points (ROP) using dynamic statistical formulas.
   - Dynamic Categorization: The system now automatically detects categories (e.g., PHARMA, PPE) from any uploaded CSV and processes them individually.

3. Multi-Config Support & Uploads:
   - File Uploading: Admins can now upload both CSV/Parquet datasets and Lead Time JSON configs directly through the UI.
   - Lead Time Management: Ability to switch between different .json configuration files to simulate various supply chain scenarios.
   - Session-Persistence: The app remembers both your Active CSV and your Active JSON Config across all pages.

4. Frontend & UI (Dark Mode):
   - dashboard.html: Updated with a dual-selection panel for data and supply chain configs.
   - instructions.html: Comprehensive guide for users on CSV headers and JSON schema.
   - risk.html / forecast.html: Full results tables with automated calculations.

5. Testing Data:
   - You can script your own CSV generators or use the provided demo data to test specific warehouse scenarios (e.g., medical spikes or holiday demand).

What Needs to Be Done:
----------------------
- Visualization: Implement charts (Chart.js or Plotly) for the Forecast page to show demand trends.
- Enhanced Error Handling: Add more robust try-except blocks for corrupted CSV uploads.
- UI Polish: Add loading spinners for the AI model inference on large datasets.

How to Run (Detailed Instructions):
----------------------------------
NOTE: Use Git Bash for all commands. If using CMD, path slashes and activation commands will differ.

1. Install Git LFS (CRITICAL):
   The LightGBM model (.pkl) is stored via Git LFS. You MUST have LFS installed to pull the actual model file.
   Download from: https://git-lfs.github.com/
   Then run in your terminal: git lfs install

2. Clone repo & Pull Model:
   git clone <repo-url>
   cd ICT304_Project
   git lfs pull

3. Create and activate virtual environment:
   python -m venv venv
   source venv/Scripts/activate

4. Install dependencies:
   pip install -r requirements.txt

5. Configure and Run Flask:
   run the command below when you are in the directory
   where app.py is located
   flask run

6. Open browser:
   Go to http://127.0.0.1:5000

USERNAME AND PASSWORDS FOR ROLES ARE LOCATED IN users.csv IN BACKEND DIRECTORY

Notes:
------
- Git LFS: If your 'wms_lgbm_model.pkl' is only a few KB in size, you didn't run 'git lfs pull'. The app will crash without the full model file.
- Folder Structure: 
    - /backend/uploads: Stores user-uploaded CSVs and custom JSONs (soon not yet implemented). (Git-ignored)
    - /backend/model: Stores the .pkl model and the default lead_times.json.
- Role-based login: Admin can upload/config; Analyst/Viewer have restricted views.
- Category Names: The cat_id in your CSV must match the keys in your lead_times.json for the Risk Analysis to calculate correctly.
