
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Load credentials from .env
load_dotenv()

# Read database configuration
db_url = URL.create(
    drivername="postgresql+psycopg",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME"),
)

# Create database engine
engine = create_engine(
    db_url,
    pool_pre_ping=True
)

# Test database connection
try:
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), version();")
        )

        database, version = result.fetchone()

        print("Connection successful!")
        print("Database:", database)
        print("PostgreSQL:", version)

except Exception as error:
    print("Database connection failed!")
    print("Error:", error)

finally:
    engine.dispose()
