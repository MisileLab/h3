"""
PostgreSQL database connection and models
"""

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, DateTime, Boolean, Text, JSON, Index
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Event(Base):
    """Event model for storing user behavior events"""
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    type: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    data: Mapped[Dict[str, Any]] = mapped_column(JSON)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    uploaded: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_type_timestamp', 'type', 'timestamp'),
    )


class Feedback(Base):
    """Feedback model for storing user feedback on predictions"""
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    event_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    feedback_type: Mapped[str] = mapped_column(String, index=True)
    prediction_id: Mapped[Optional[str]] = mapped_column(String)
    selected_index: Mapped[Optional[int]] = mapped_column(Integer)
    context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    content: Mapped[Optional[str]] = mapped_column(Text)
    rating: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    __table_args__ = (
        Index('idx_feedback_user_created', 'user_id', 'created_at'),
        Index('idx_feedback_type_created', 'feedback_type', 'created_at'),
    )


class ModelVersion(Base):
    """Model version tracking"""
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    file_path: Mapped[Optional[str]] = mapped_column(String)
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    checksum: Mapped[Optional[str]] = mapped_column(String)
    accuracy_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    training_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    download_count: Mapped[int] = mapped_column(Integer, default=0)


class UserSession(Base):
    """User session tracking"""
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    event_count: Mapped[int] = mapped_column(Integer, default=0)
    domain_count: Mapped[int] = mapped_column(Integer, default=0)
    session_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    __table_args__ = (
        Index('idx_session_user_started', 'user_id', 'started_at'),
    )


class Database:
    """Database connection and operations manager"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or "postgresql+asyncpg://nextpred:password@localhost:5432/nextpred"
        self.engine = None
        self.session_factory = None
        self.pool = None

    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")

    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        return self.session_factory()

    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Get database status information"""
        try:
            async with self.get_session() as session:
                # Get table counts
                events_count = await session.execute("SELECT COUNT(*) FROM events")
                feedback_count = await session.execute("SELECT COUNT(*) FROM feedback")
                models_count = await session.execute("SELECT COUNT(*) FROM model_versions")

                return {
                    "status": "healthy",
                    "events_count": events_count.scalar(),
                    "feedback_count": feedback_count.scalar(),
                    "models_count": models_count.scalar(),
                    "connection_pool": {
                        "size": self.engine.pool.size() if self.engine.pool else 0,
                        "checked_in": self.engine.pool.checkedin() if self.engine.pool else 0,
                        "checked_out": self.engine.pool.checkedout() if self.engine.pool else 0,
                    }
                }
        except Exception as e:
            logger.error(f"Failed to get database status: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def store_event(self, event_data: Dict[str, Any]) -> str:
        """Store a single event"""
        try:
            async with self.get_session() as session:
                event = Event(
                    id=event_data.get("id", self._generate_id()),
                    user_id=event_data.get("user_id"),
                    type=event_data["type"],
                    timestamp=event_data["timestamp"],
                    data=event_data["data"]
                )

                session.add(event)
                await session.commit()
                await session.refresh(event)

                return event.id

        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            raise

    async def store_events_batch(self, events: List[Dict[str, Any]]) -> int:
        """Store multiple events in batch"""
        try:
            async with self.get_session() as session:
                event_objects = []
                for event_data in events:
                    event = Event(
                        id=event_data.get("id", self._generate_id()),
                        user_id=event_data.get("user_id"),
                        type=event_data["type"],
                        timestamp=event_data["timestamp"],
                        data=event_data["data"]
                    )
                    event_objects.append(event)

                session.add_all(event_objects)
                await session.commit()

                return len(event_objects)

        except Exception as e:
            logger.error(f"Failed to store events batch: {e}")
            raise

    async def get_events_by_user(
        self, 
        user_id: str, 
        limit: int = 1000,
        offset: int = 0
    ) -> List[Event]:
        """Get events for a specific user"""
        try:
            async with self.get_session() as session:
                from sqlalchemy import select

                stmt = (
                    select(Event)
                    .where(Event.user_id == user_id)
                    .order_by(Event.timestamp.desc())
                    .limit(limit)
                    .offset(offset)
                )

                result = await session.execute(stmt)
                return result.scalars().all()

        except Exception as e:
            logger.error(f"Failed to get events for user {user_id}: {e}")
            raise

    async def get_recent_events(
        self, 
        minutes: int = 10,
        user_id: Optional[str] = None
    ) -> List[Event]:
        """Get recent events within time window"""
        try:
            async with self.get_session() as session:
                from sqlalchemy import select
                import time

                cutoff_time = int(time.time() * 1000) - (minutes * 60 * 1000)

                stmt = select(Event).where(Event.timestamp > cutoff_time)

                if user_id:
                    stmt = stmt.where(Event.user_id == user_id)

                stmt = stmt.order_by(Event.timestamp.desc())

                result = await session.execute(stmt)
                return result.scalars().all()

        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            raise

    async def store_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Store user feedback"""
        try:
            async with self.get_session() as session:
                feedback = Feedback(
                    id=self._generate_id(),
                    user_id=feedback_data.get("user_id"),
                    event_id=feedback_data.get("event_id"),
                    feedback_type=feedback_data["feedback_type"],
                    prediction_id=feedback_data.get("prediction_id"),
                    selected_index=feedback_data.get("selected_index"),
                    context=feedback_data.get("context"),
                    content=feedback_data.get("content"),
                    rating=feedback_data.get("rating"),
                    metadata=feedback_data.get("metadata")
                )

                session.add(feedback)
                await session.commit()
                await session.refresh(feedback)

                return feedback.id

        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            raise

    async def get_model_version(self, version: str = None) -> Optional[ModelVersion]:
        """Get model version information"""
        try:
            async with self.get_session() as session:
                from sqlalchemy import select

                if version:
                    stmt = select(ModelVersion).where(ModelVersion.version == version)
                else:
                    stmt = (
                        select(ModelVersion)
                        .where(ModelVersion.is_active == True)
                        .order_by(ModelVersion.created_at.desc())
                    )

                result = await session.execute(stmt)
                return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            raise

    async def store_model_version(self, model_data: Dict[str, Any]) -> str:
        """Store model version information"""
        try:
            async with self.get_session() as session:
                # Deactivate previous versions
                await session.execute(
                    "UPDATE model_versions SET is_active = false WHERE is_active = true"
                )

                model_version = ModelVersion(
                    id=self._generate_id(),
                    version=model_data["version"],
                    file_path=model_data.get("file_path"),
                    file_size=model_data.get("file_size"),
                    checksum=model_data.get("checksum"),
                    accuracy_metrics=model_data.get("accuracy_metrics"),
                    training_config=model_data.get("training_config"),
                    is_active=True
                )

                session.add(model_version)
                await session.commit()
                await session.refresh(model_version)

                return model_version.id

        except Exception as e:
            logger.error(f"Failed to store model version: {e}")
            raise

    def _generate_id(self) -> str:
        """Generate unique ID"""
        import time
        import random
        return f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


# Dependency function for FastAPI
async def get_db() -> Database:
    """Get database instance"""
    # This would typically be initialized at app startup
    # For now, return a new instance
    return Database()