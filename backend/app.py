from flask import Flask

from config import Config
from routes import routes


def create_app():

    app = Flask(__name__, template_folder=Config.template_dir)

    app.config.from_object(Config)

    app.register_blueprint(routes)

    return app


if __name__ == "__main__":

    app = create_app()

    app.run(debug=True)

