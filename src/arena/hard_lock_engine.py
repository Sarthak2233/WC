"""
hard_lock_engine.py — Temporal State Machine with Immutable Append-Only Ledger.

Per CORE_PROBLEM_TO_SOLVE.md:
  - Once a match deadline passes, predictions are hard-locked. No rollback.
  - Once a result is ingested, it is immutable.
  - All state transitions are persisted to append-only CSV ledgers.
"""
import csv
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

LOCK_LEDGER_PATH = "data/arena/lock_ledger.csv"
RESULTS_LEDGER_PATH = "data/arena/results_ledger.csv"

_LOCK_FIELDS = ["match_id", "locked_at"]
_RESULT_FIELDS = ["match_id", "home_goals", "away_goals", "stage", "ingested_at"]


class LockViolationError(Exception):
    """Raised when attempting to lock an already-locked match."""


class ResultAlreadyExistsError(Exception):
    """Raised when attempting to ingest a result for an already-scored match."""


class MatchNotLockedError(Exception):
    """Raised when trying to ingest a result for a match not yet locked."""


@dataclass(frozen=True)
class MatchResult:
    """Immutable match result record.

    Attributes:
        match_id: Integer match identifier.
        home_goals: Final home goals (90 min, excluding penalties).
        away_goals: Final away goals.
        stage: Tournament stage name.
        ingested_at: UTC timestamp of ingestion.
    """

    match_id: int
    home_goals: int
    away_goals: int
    stage: str
    ingested_at: datetime


class HardLockEngine:
    """Temporal State Machine managing match locking and result ingestion.

    Both the lock ledger and results ledger are append-only CSV files.
    On init, existing state is hydrated into in-memory sets for O(1) lookups.

    Args:
        lock_path: Path to the lock ledger CSV.
        results_path: Path to the results ledger CSV.
    """

    def __init__(
        self,
        lock_path: str = LOCK_LEDGER_PATH,
        results_path: str = RESULTS_LEDGER_PATH,
    ) -> None:
        self._lock_path = lock_path
        self._results_path = results_path
        # In-memory O(1) lookup sets
        self._locked_matches: Set[int] = set()
        self._ingested_results: Dict[int, MatchResult] = {}

        self._hydrate()

    def _hydrate(self) -> None:
        """Load existing lock and result ledgers into memory (O(N))."""
        if os.path.isfile(self._lock_path):
            with open(self._lock_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._locked_matches.add(int(row["match_id"]))
            logger.info("Hydrated %d lock records.", len(self._locked_matches))

        if os.path.isfile(self._results_path):
            with open(self._results_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    mid = int(row["match_id"])
                    self._ingested_results[mid] = MatchResult(
                        match_id=mid,
                        home_goals=int(row["home_goals"]),
                        away_goals=int(row["away_goals"]),
                        stage=row["stage"],
                        ingested_at=datetime.fromisoformat(row["ingested_at"]),
                    )
            logger.info("Hydrated %d result records.", len(self._ingested_results))

    def _append_to_csv(self, path: str, fieldnames: list, row: dict) -> None:
        """Append a single row to a CSV, creating the file/header if needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def lock_match(
        self,
        match_id: int,
        kickoff_time: Optional[datetime] = None,
    ) -> None:
        """Lock a match, preventing further predictions.

        Args:
            match_id: Match to lock.
            kickoff_time: If provided, raises an error if utcnow() < kickoff_time
                         (i.e., the match hasn't started yet). If None, locks immediately.

        Raises:
            LockViolationError: If the match is already locked.
            ValueError: If kickoff_time is in the future.
        """
        if match_id in self._locked_matches:
            raise LockViolationError(
                f"Match {match_id} is already locked. No rollback permitted."
            )

        now = datetime.now(timezone.utc)
        if kickoff_time is not None and now < kickoff_time:
            raise ValueError(
                f"Cannot lock match {match_id}: kickoff is in the future ({kickoff_time})."
            )

        # Persist to immutable ledger FIRST, then update memory
        locked_at = now.isoformat()
        self._append_to_csv(
            self._lock_path,
            _LOCK_FIELDS,
            {"match_id": match_id, "locked_at": locked_at},
        )
        self._locked_matches.add(match_id)
        logger.info("Match %d hard-locked at %s.", match_id, locked_at)

    def is_locked(self, match_id: int) -> bool:
        """Return whether a match is locked (O(1))."""
        return match_id in self._locked_matches

    def ingest_result(
        self,
        match_id: int,
        home_goals: int,
        away_goals: int,
        stage: str,
    ) -> MatchResult:
        """Ingest an actual match result into the immutable results ledger.

        Args:
            match_id: Match that has just finished.
            home_goals: Actual home goals scored.
            away_goals: Actual away goals scored.
            stage: Tournament stage name.

        Returns:
            The MatchResult dataclass for downstream processing.

        Raises:
            MatchNotLockedError: If the match was never locked.
            ResultAlreadyExistsError: If a result already exists for this match.
        """
        if match_id not in self._locked_matches:
            raise MatchNotLockedError(
                f"Match {match_id} must be locked before result ingestion."
            )

        if match_id in self._ingested_results:
            raise ResultAlreadyExistsError(
                f"Result for match {match_id} already ingested. Ledger is immutable."
            )

        now = datetime.now(timezone.utc)
        result = MatchResult(
            match_id=match_id,
            home_goals=home_goals,
            away_goals=away_goals,
            stage=stage,
            ingested_at=now,
        )

        # Persist to immutable ledger FIRST
        self._append_to_csv(
            self._results_path,
            _RESULT_FIELDS,
            {
                "match_id": match_id,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "stage": stage,
                "ingested_at": now.isoformat(),
            },
        )
        self._ingested_results[match_id] = result
        logger.info(
            "Result ingested for match %d: %d-%d (%s).",
            match_id, home_goals, away_goals, stage,
        )
        return result

    def get_result(self, match_id: int) -> Optional[MatchResult]:
        """Retrieve an ingested result by match_id (O(1))."""
        return self._ingested_results.get(match_id)
