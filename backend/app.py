from flask import Flask

from config import Config
from routes import routes


def create_app():

    app = Flask(__name__, template_folder=Config.template_dir)

    app.config.from_object(Config)

    app.register_blueprint(routes)

    print(f"Template folder: {app.template_folder}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Demo data: {app.config.get('DEMO_DATA_PATH', 'NOT SET')}")



    return app


if __name__ == "__main__":

    app = create_app()

    app.run(debug=True)

