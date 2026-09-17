"""Database base configuration and engine creation."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase


class Base(AsyncAttrs, DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass
