from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///models.db"
engine = create_engine(DATABASE_URL)

Session = sessionmaker(bind=engine)

def init_db():
    from models.model_metadata import Base
    Base.metadata.create_all(engine)