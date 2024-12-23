from sqlalchemy import Column, String, Float, Enum, Integer
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum as PyEnum

Base = declarative_base()

class ModelStatus(PyEnum):
    IN_PROGRESS = "IN_PROGRESS"
    READY = "READY"
    FAILED = "FAILED"

class ModelMetadata(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_name = Column(String, unique=True, nullable=False)
    feature = Column(String, nullable=False)
    model_score = Column(Float, nullable=True)
    status = Column(Enum(ModelStatus), default=ModelStatus.IN_PROGRESS, nullable=False)