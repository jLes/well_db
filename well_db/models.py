"""
Database models, static lookup tables, and Pydantic schemas for NM OCD Wells.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Float, DateTime, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# Field Mappings: Webpage Labels -> Database Field Names
# This is the single source of truth for mapping scraped webpage labels
# to our internal database field names.
WEBPAGE_FIELD_MAPPINGS: dict[str, str] = {
    # General well information
    "Operator:": "operator",
    "Status:": "status",
    "Well Type:": "well_type",
    "Work Type:": "work_type",
    "Direction:": "directional_status",
    "Multi-Lateral:": "multi_lateral",
    # Ownership
    "Mineral Owner:": "mineral_owner",
    "Surface Owner:": "surface_owner",
    # Location
    "Surface Location:": "surface_location",
    # Elevation
    "GL Elevation:": "gl_elevation",
    "KB Elevation:": "kb_elevation",
    "DF Elevation:": "df_elevation",
    # Well details
    "Sing/Mult Compl:": "single_multi_completion",
    "Potash Waiver:": "potash_waiver",
    "True Vertical Depth:": "tvd",
    # Dates (note: webpage uses "Spud:" not "Spud Date:")
    "Spud:": "spud_date",
    "Last Inspection:": "last_inspection",
}