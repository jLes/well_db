"""
Database models, static lookup tables, and Pydantic schemas for NM OCD Wells.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Float, DateTime, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_CSV_PATH = PROJECT_ROOT / "resources" / "apis_pythondev_test.csv"


# Field Mappings: Webpage Labels -> Database Field Names
# This is the single source of truth for mapping scraped webpage labels
# to our internal database field names.
# Order matches the spec: Operator, Status, Well Type, Work Type, Directional Status,
# Multi-Lateral, Mineral Owner, Surface Owner, Surface Location, GL Elevation,
# KB Elevation, DF Elevation, Single/Multiple Completion, Potash Waiver, Spud Date,
# Last Inspection, TVD, API, Latitude, Longitude, CRS
WEBPAGE_FIELD_MAPPINGS: dict[str, str] = {
    "Operator:": "operator",
    "Status:": "status",
    "Well Type:": "well_type",
    "Work Type:": "work_type",
    "Direction:": "directional_status",
    "Multi-Lateral:": "multi_lateral",
    "Mineral Owner:": "mineral_owner",
    "Surface Owner:": "surface_owner",
    "Surface Location:": "surface_location",
    "GL Elevation:": "gl_elevation",
    "KB Elevation:": "kb_elevation",
    "DF Elevation:": "df_elevation",
    "Sing/Mult Compl:": "single_multi_completion",
    "Potash Waiver:": "potash_waiver",
    "Spud:": "spud_date",
    "Last Inspection:": "last_inspection",
    "True Vertical Depth:": "tvd",
    # API, Latitude, Longitude, CRS are handled separately (not label-based extraction)
}


class WellData(Base):
    """
    SQLAlchemy model for the api_well_data table.

    Column order: API first (Moved to first column bc primary key), then spec order for remaining fields.
    Spec order: Operator, Status, Well Type, Work Type, Directional Status,
    Multi-Lateral, Mineral Owner, Surface Owner, Surface Location, GL Elevation,
    KB Elevation, DF Elevation, Single/Multiple Completion, Potash Waiver, Spud Date,
    Last Inspection, TVD, Latitude, Longitude, CRS
    """

    __tablename__ = "api_well_data"

    # Primary key first
    api = Column(String, primary_key=True, index=True)

    # Remaining columns per spec order
    operator = Column(String, nullable=True)
    status = Column(String, nullable=True)
    well_type = Column(String, nullable=True)
    work_type = Column(String, nullable=True)
    directional_status = Column(String, nullable=True)
    multi_lateral = Column(String, nullable=True)
    mineral_owner = Column(String, nullable=True)
    surface_owner = Column(String, nullable=True)
    surface_location = Column(String, nullable=True)
    gl_elevation = Column(String, nullable=True)
    kb_elevation = Column(String, nullable=True)
    df_elevation = Column(String, nullable=True)
    single_multi_completion = Column(String, nullable=True)
    potash_waiver = Column(String, nullable=True)
    spud_date = Column(String, nullable=True)
    last_inspection = Column(String, nullable=True)
    tvd = Column(String, nullable=True)  # True Vertical Depth
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    crs = Column(String, nullable=True)

    # Metadata (not in spec, for internal tracking)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Composite index for geospatial queries
    __table_args__ = (
        Index('idx_lat_lon', 'latitude', 'longitude'),
    )

    def to_dict(self) -> dict:
        """Convert model to dictionary, API first then spec order."""
        return {
            "api": self.api,
            "operator": self.operator,
            "status": self.status,
            "well_type": self.well_type,
            "work_type": self.work_type,
            "directional_status": self.directional_status,
            "multi_lateral": self.multi_lateral,
            "mineral_owner": self.mineral_owner,
            "surface_owner": self.surface_owner,
            "surface_location": self.surface_location,
            "gl_elevation": self.gl_elevation,
            "kb_elevation": self.kb_elevation,
            "df_elevation": self.df_elevation,
            "single_multi_completion": self.single_multi_completion,
            "potash_waiver": self.potash_waiver,
            "spud_date": self.spud_date,
            "last_inspection": self.last_inspection,
            "tvd": self.tvd,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "crs": self.crs,
        }
