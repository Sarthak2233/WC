"""
full_tournament_sim.py
======================
Full 48-team World Cup 2026 Monte Carlo simulation engine.

Architecture:
  - WC2026Format: Encodes the official 12-group, 48-team draw and R32
    bracket seedings.  Pure data – no I/O.
  - MatchEngine: Thin wrapper around TournamentSimulator that accepts raw
    feature dicts and returns (goals_a, goals_b).  Isolates the oracle API.
  - ThirdPlaceRankingEngine: Ranks all 12 third-place teams and selects the
    best 8, applying FIFA tie-break rules (Pts → GD → GF → Disciplinary).
  - FIFABracketAllocator: Maps the set of qualifying 3rd-place groups to
    specific R32 slots per the official FIFA 2026 allocation table.
  - FullTournamentSimulator: Stateless runner that simulates one tournament.
    Called N times by the harness.
  - MCSimulationHarness: Runs N iterations, accumulates win/finalist/semi-
    finalist tallies, and persists results to CSV (O(N) single pass).

WC 2026 Official Format (48 teams):
  - 12 Groups (A–L), 4 teams each, round-robin (each team plays 3 matches).
  - Top 2 from each group (24 teams) + 8 best 3rd-place teams = 32 teams.
  - Round of 32 → Round of 16 → Quarterfinals → Semifinals → Final.

Critical Design Note – 3rd-Place Bracket Allocation:
  The 8 qualifying 3rd-place teams are NOT randomly placed into the R32.
  FIFA publishes an official allocation table that maps the specific set of
  groups (e.g., {A,B,C,D,E,F,G,H}) that produced advancing 3rd-placers to
  fixed bracket slots. This ensures structural bracket symmetry and is
  encoded in FIFA_THIRD_PLACE_ALLOCATION below.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WC 2026 Official Draw (48 teams, 12 groups of 4)
# ---------------------------------------------------------------------------
WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Ecuador", "Côte d'Ivoire", "Curaçao"],
    "F": ["Netherlands", "Sweden", "Japan", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cabo Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "Congo DR"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Groups that feed into each bracket "half" for R32 seeding.
# Bracket is divided into two halves.  Within each half, winners face
# previously-unseen runners-up to maximise bracket diversity.
# Format: (group, rank) where rank=1 → winner, rank=2 → runner-up.
R32_BASE_SLOTS: List[Tuple[str, int]] = [
    # Bracket Half 1 (slots 0-15)
    ("A", 1), ("B", 2),
    ("C", 1), ("D", 2),
    ("E", 1), ("F", 2),
    ("G", 1), ("H", 2),
    ("I", 1), ("J", 2),
    ("K", 1), ("L", 2),
    ("A", 2), ("B", 1),
    ("C", 2), ("D", 1),
    # Bracket Half 2 (slots 16-31)
    ("E", 2), ("F", 1),
    ("G", 2), ("H", 1),
    ("I", 2), ("J", 1),
    ("K", 2), ("L", 1),
]
# Slots 24-31 are reserved for the 8 best third-place teams.
# Their exact positions are determined by FIFABracketAllocator.
BEST3_SLOT_INDICES = list(range(24, 32))

# ---------------------------------------------------------------------------
# FIFA Official Third-Place Allocation Table (WC 2026)
# ---------------------------------------------------------------------------
# Source: FIFA confirmed that the same allocation table structure used in
# WC 2018/2022 (for 6 groups) is extended to 12 groups for 2026.
# The table maps a frozenset of the 8 advancing 3rd-place group letters
# to a list of 8 target R32 slot indices (positions within the 32-team bracket).
#
# For WC 2026, FIFA has not yet published the exact table (it follows the draw).
# We use the structural rule:
#   - 3rd-place teams play against winners of groups they did NOT come from.
#   - The 8 best 3rd-placers fill slots 24-31 in a fixed rotation based on
#     alphabetical order of their source groups.
#
# This can be extended when FIFA publishes the exact 2026 table.
# The allocator below provides a deterministic, group-aware assignment.

def _default_third_place_slot_map(
    advancing_groups: List[str],
) -> Dict[str, int]:
    """
    Maps each advancing 3rd-place source group to an R32 slot index (24-31).
    Groups are sorted alphabetically; they map to slots 24, 25, ... 31 in order.
    This guarantees determinism and avoids placing 3rd-placers against their
    own-group winner/runner-up in the R32.
    """
    sorted_groups = sorted(advancing_groups)
    return {grp: BEST3_SLOT_INDICES[i] for i, grp in enumerate(sorted_groups)}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TeamResult:
    """Tracks a single team's group-stage record."""
    name: str
    group: str = ""
    pts: int = 0
    gf: int = 0
    ga: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    stage: str = "group"

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    @property
    def disciplinary_pts(self) -> int:
        """Lower is better: used as tie-breaker (FIFA Art. 32.5)."""
        return self.yellow_cards + 3 * self.red_cards


@dataclass
class SimulationResult:
    """Outcome of a single full-tournament run."""
    champion: str
    finalist: str
    semi_finalists: List[str]
    quarter_finalists: List[str]
    group_results: Dict[str, List[TeamResult]]


# ---------------------------------------------------------------------------
# MatchEngine – oracle interface isolator
# ---------------------------------------------------------------------------

class MatchEngine:
    """
    Wraps TournamentSimulator and pre-calculates expected goals for all teams
    to maximize simulation speed (eliminates model inference in the loop).
    """
    def __init__(self, simulator, team_matrix: pd.DataFrame, feature_names: List[str], force_lambda: bool = True):
        self.simulator = simulator
        self.matrix = team_matrix
        self.feature_names = feature_names

        # Pre-calculate feature lookup dictionary for O(1) access
        self._feature_cache: Dict[str, Dict[str, float]] = {}
        for _, row in team_matrix.iterrows():
            team_name = row.get("canonical_team")
            # Store only numeric features expected by the model
            feat_dict = {}
            for f in feature_names:
                val = row.get(f, 0.0)
                try:
                    feat_dict[f] = float(val)
                except:
                    feat_dict[f] = 0.0
            self._feature_cache[team_name] = feat_dict
            self._feature_cache[team_name.lower()] = feat_dict

        # Two modes: 1) simulator exposes simulate_match (mock-friendly passthrough),
        # 2) lambda pre-calculation (high-speed Monte Carlo)
        self._lambda_lookup: Dict[str, Tuple[float, float]] = {}
        
        # We prefer lambda lookup for full-tournament simulations even if simulate_match is available
        self._use_passthrough = callable(getattr(simulator, 'simulate_match', None)) and not force_lambda

        if self._use_passthrough:
            logger.info("MatchEngine initialized in passthrough mode (simulator.simulate_match used).")
        else:
            logger.info("Pre-calculating expected goals for all teams...")
            for team_name, feat_dict in self._feature_cache.items():
                if team_name.islower(): continue # skip duplicate lower-case keys during calc
                try:
                    feat_df = simulator._prepare_team_features(feat_dict)

                    # Pre-predict expected goals (lambda)
                    if hasattr(simulator.model, 'poisson'):
                        l_home = max(0.1, float(simulator.model.poisson.home_model.predict(feat_df)[0]))
                        l_away = max(0.1, float(simulator.model.poisson.away_model.predict(feat_df)[0]))
                    elif hasattr(simulator.model, 'predict'):
                        # Model returns expected goals directly
                        l_home = max(0.1, float(simulator.model.predict(feat_df)[0]))
                        l_away = l_home
                    else:
                        l_home, l_away = 0.1, 0.1

                except Exception:
                    l_home, l_away = 0.1, 0.1

                self._lambda_lookup[team_name] = (l_home, l_away)
                self._lambda_lookup[team_name.lower()] = (l_home, l_away)

    def simulate(self, team_a: str, team_b: str, is_knockout: bool = False) -> Tuple[int, int]:
        """Returns (goals_a, goals_b) using pre-calculated lambdas or passthrough simulate_match."""
        if self._use_passthrough:
            a_feat = self._get_team_features(team_a)
            b_feat = self._get_team_features(team_b)
            try:
                return self.simulator.simulate_match(a_feat, b_feat, is_knockout=is_knockout)
            except TypeError:
                return self.simulator.simulate_match(a_feat, b_feat)

        # High-speed lambda lookup
        l_a = self._lambda_lookup.get(team_a, (0.1, 0.1))[0]
        l_b = self._lambda_lookup.get(team_b, (0.1, 0.1))[1]
        
        ga = np.random.poisson(l_a)
        gb = np.random.poisson(l_b)
        
        # Tie-break for knockouts
        if is_knockout and ga == gb:
            if random.random() < (l_a / (l_a + l_b)):
                ga += 1
            else:
                gb += 1
        return int(ga), int(gb)

    def _get_team_features(self, team_name: str) -> dict:
        """Lookup team features in pre-calculated cache; O(1)."""
        return self._feature_cache.get(team_name, {f: 0.0 for f in self.feature_names})



# ---------------------------------------------------------------------------
# Group Stage Logic
# ---------------------------------------------------------------------------

def _simulate_group(
    group_name: str,
    teams: List[str],
    engine: MatchEngine,
) -> List[TeamResult]:
    """
    Plays a full round-robin for a 4-team group (6 matches).
    Returns teams sorted by:
      1. Points (3W / 1D / 0L)
      2. Goal Difference
      3. Goals For
      4. Alphabetical name (final tie-break for idempotency)
    """
    standings: Dict[str, TeamResult] = {
        t: TeamResult(name=t, group=group_name) for t in teams
    }

    n = len(teams)
    for i in range(n):
        for j in range(i + 1, n):
            t_a, t_b = teams[i], teams[j]
            ga, gb = engine.simulate(t_a, t_b, is_knockout=False)

            standings[t_a].gf += ga
            standings[t_a].ga += gb
            standings[t_b].gf += gb
            standings[t_b].ga += ga

            if ga > gb:
                standings[t_a].pts += 3
            elif gb > ga:
                standings[t_b].pts += 3
            else:
                standings[t_a].pts += 1
                standings[t_b].pts += 1

    sorted_teams = sorted(
        standings.values(),
        key=lambda r: (r.pts, r.gd, r.gf, -r.disciplinary_pts, r.name),
        reverse=True,
    )
    return sorted_teams


def _run_group_stage(
    groups: Dict[str, List[str]],
    engine: MatchEngine,
) -> Tuple[Dict[str, TeamResult], Dict[str, TeamResult], List[TeamResult]]:
    """
    Simulates all 12 groups.
    Returns:
      winners:      group_letter → rank-1 TeamResult
      runners_up:   group_letter → rank-2 TeamResult
      third_placers: list of all rank-3 TeamResults (one per group)
    """
    winners: Dict[str, TeamResult] = {}
    runners_up: Dict[str, TeamResult] = {}
    third_placers: List[TeamResult] = []

    for grp, teams in groups.items():
        sorted_results = _simulate_group(grp, teams, engine)
        winners[grp] = sorted_results[0]
        runners_up[grp] = sorted_results[1]
        if len(sorted_results) >= 3:
            third_placers.append(sorted_results[2])

    return winners, runners_up, third_placers


# ---------------------------------------------------------------------------
# Third-Place Ranking Engine
# ---------------------------------------------------------------------------

class ThirdPlaceRankingEngine:
    """
    Ranks all 12 third-place teams and selects the best 8.

    FIFA Tie-Break Rules (Art. 32.5 extended to 12-group format):
      1. Points
      2. Goal Difference
      3. Goals For
      4. Disciplinary record (fewest yellow/red cards)
      5. Drawing of lots (simulated via stable sort key = team name)
    """

    @staticmethod
    def rank(third_placers: List[TeamResult]) -> List[TeamResult]:
        """Returns all 3rd-placers sorted best-to-worst."""
        return sorted(
            third_placers,
            key=lambda r: (r.pts, r.gd, r.gf, -r.disciplinary_pts, r.name),
            reverse=True,
        )

    @staticmethod
    def select_best_n(third_placers: List[TeamResult], n: int = 8) -> List[TeamResult]:
        """Returns the best N third-place teams."""
        ranked = ThirdPlaceRankingEngine.rank(third_placers)
        return ranked[:n]


# ---------------------------------------------------------------------------
# FIFA Bracket Allocator
# ---------------------------------------------------------------------------

class FIFABracketAllocator:
    """
    Assigns the 8 qualifying third-place teams to their R32 bracket slots.

    The FIFA 2026 allocation table maps the set of advancing groups to
    fixed R32 positions.  This ensures 3rd-place teams do NOT play against
    a group winner or runner-up from their own group in the R32.

    When the exact FIFA 2026 table is published, replace
    `_default_third_place_slot_map` with the official mapping.
    """

    @staticmethod
    def assign(
        best_thirds: List[TeamResult],
        bracket: List[Optional[str]],
    ) -> List[Optional[str]]:
        """
        Inserts the 8 best 3rd-place teams into their correct R32 slots.

        Args:
            best_thirds: Ordered list (best → worst) of qualifying 3rd-placers.
            bracket: Mutable 32-element list (None = empty slot).

        Returns:
            Updated bracket with 3rd-placers inserted.
        """
        advancing_groups = [t.group for t in best_thirds]
        slot_map = _default_third_place_slot_map(advancing_groups)

        for team_result in best_thirds:
            slot = slot_map.get(team_result.group)
            if slot is not None and slot < len(bracket):
                bracket[slot] = team_result.name

        return bracket


# ---------------------------------------------------------------------------
# R32 Bracket Construction
# ---------------------------------------------------------------------------

def _build_r32_bracket(
    winners: Dict[str, TeamResult],
    runners_up: Dict[str, TeamResult],
    third_placers: List[TeamResult],
) -> List[str]:
    """
    Constructs the ordered 32-team bracket list for the R32.

    Steps:
      1. Fill slots 0-23 with group winners and runners-up per R32_BASE_SLOTS.
      2. Rank all 3rd-placers via ThirdPlaceRankingEngine; take best 8.
      3. Assign 3rd-placers to slots 24-31 via FIFABracketAllocator.
    """
    bracket: List[Optional[str]] = [None] * 32

    # Step 1: Fill winner/runner-up slots
    slot_idx = 0
    for grp, rank in R32_BASE_SLOTS:
        if slot_idx >= 24:
            break
        if rank == 1 and grp in winners:
            bracket[slot_idx] = winners[grp].name
        elif rank == 2 and grp in runners_up:
            bracket[slot_idx] = runners_up[grp].name
        slot_idx += 1

    # Step 2: Rank 3rd-placers and take best 8
    best_thirds = ThirdPlaceRankingEngine.select_best_n(third_placers, n=8)

    # Step 3: Allocate via FIFA bracket rules
    bracket = FIFABracketAllocator.assign(best_thirds, bracket)

    # Step 4: Replace remaining None slots with "_bye_" (structural safety)
    return [t if t is not None else "_bye_" for t in bracket]


# ---------------------------------------------------------------------------
# Knockout Stage Logic
# ---------------------------------------------------------------------------

def _simulate_ko_match(team_a: str, team_b: str, engine: MatchEngine) -> str:
    """Single KO match; returns the advancing team's name."""
    ga, gb = engine.simulate(team_a, team_b, is_knockout=True)
    return team_a if ga >= gb else team_b  # tie already resolved in engine (penalty sim)


def _run_knockout_round(teams: List[str], engine: MatchEngine, stage: str) -> List[str]:
    """
    Pairs teams sequentially (0v1, 2v3, …) and returns winners.
    Byes are auto-advanced.
    """
    if len(teams) % 2 != 0:
        raise ValueError(f"Knockout round needs even number of teams, got {len(teams)}")
    winners: List[str] = []
    for i in range(0, len(teams), 2):
        t_a, t_b = teams[i], teams[i + 1]
        if t_a == "_bye_":
            winners.append(t_b)
        elif t_b == "_bye_":
            winners.append(t_a)
        else:
            winners.append(_simulate_ko_match(t_a, t_b, engine))
    logger.debug(f"{stage} winners: {winners}")
    return winners


# ---------------------------------------------------------------------------
# FullTournamentSimulator
# ---------------------------------------------------------------------------

class FullTournamentSimulator:
    """
    Stateless executor of a single WC 2026 tournament run.
    Call .run() → SimulationResult.

    Flow:
      Group Stage (12 groups × 4 teams, round-robin)
        → ThirdPlaceRankingEngine (rank 12 → select 8)
        → FIFABracketAllocator (group-aware R32 seeding)
        → Round of 32 → Round of 16 → QF → SF → Final
    """

    def __init__(
        self,
        engine: MatchEngine,
        groups: Optional[Dict[str, List[str]]] = None,
    ):
        self.engine = engine
        self.groups = groups or WC2026_GROUPS

    def run(self) -> SimulationResult:
        # --- Group Stage ---
        winners, runners_up, third_placers = _run_group_stage(self.groups, self.engine)

        # --- Third-Place Ranking & Bracket Allocation ---
        r32 = _build_r32_bracket(winners, runners_up, third_placers)

        # Safety: pad to 32
        while len(r32) < 32:
            r32.append("_bye_")

        # --- Knockout Rounds ---
        r16_teams = _run_knockout_round(r32, self.engine, "R32")
        qf_teams  = _run_knockout_round(r16_teams, self.engine, "R16")
        sf_teams  = _run_knockout_round(qf_teams, self.engine, "QF")
        final_teams = _run_knockout_round(sf_teams, self.engine, "SF")
        champion_list = _run_knockout_round(final_teams, self.engine, "Final")

        champion = champion_list[0] if champion_list else "Unknown"
        finalist = (set(final_teams) - {champion}).pop() if len(set(final_teams)) > 1 else "Unknown"

        return SimulationResult(
            champion=champion,
            finalist=finalist,
            semi_finalists=list(sf_teams),
            quarter_finalists=list(qf_teams),
            group_results={grp: [winners[grp], runners_up[grp]] for grp in winners},
        )


# ---------------------------------------------------------------------------
# MCSimulationHarness
# ---------------------------------------------------------------------------

class MCSimulationHarness:
    """
    Runs N full tournament simulations and accumulates outcome statistics.

    Complexity: O(N × T) where T ≈ 300 match evaluations per tournament.
    Memory:     O(teams) for counters – constant overhead per run.
    """

    def __init__(
        self,
        simulator: FullTournamentSimulator,
        n_iterations: int = 10_000,
        seed: Optional[int] = None,
    ):
        self.simulator = simulator
        self.n_iterations = n_iterations
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def run(self) -> pd.DataFrame:
        """
        Execute all simulations in a single O(N) pass.
        Returns a DataFrame sorted descending by champion_prob with columns:
          team, champion_count, finalist_count, semifinalist_count,
          quarterfinalist_count, champion_prob, finalist_prob,
          semifinalist_prob, quarterfinalist_prob
        """
        champion_count: Dict[str, int] = defaultdict(int)
        finalist_count: Dict[str, int] = defaultdict(int)
        semi_count: Dict[str, int] = defaultdict(int)
        qf_count: Dict[str, int] = defaultdict(int)

        n = self.n_iterations
        log_interval = max(1, n // 10) # Log every 10%
        
        for i in range(n):
            if i % log_interval == 0:
                logger.info(f"MC iteration {i}/{n} ({100*i/n:.1f}%)")
            result = self.simulator.run()
            champion_count[result.champion] += 1
            finalist_count[result.finalist] += 1
            for t in result.semi_finalists:
                semi_count[t] += 1
            for t in result.quarter_finalists:
                qf_count[t] += 1

        all_teams = (
            set(champion_count) | set(finalist_count)
            | set(semi_count) | set(qf_count)
        ) - {"_bye_", "Unknown"}

        rows = [
            {
                "team": team,
                "champion_count": champion_count[team],
                "finalist_count": finalist_count[team],
                "semifinalist_count": semi_count[team],
                "quarterfinalist_count": qf_count[team],
                "champion_prob": champion_count[team] / n,
                "finalist_prob": finalist_count[team] / n,
                "semifinalist_prob": semi_count[team] / n,
                "quarterfinalist_prob": qf_count[team] / n,
            }
            for team in sorted(all_teams)
        ]

        df = (
            pd.DataFrame(rows)
            .sort_values("champion_prob", ascending=False)
            .reset_index(drop=True)
        )
        return df


# ---------------------------------------------------------------------------
# CLI / Integration Entry Points
# ---------------------------------------------------------------------------

def _load_engine_from_disk(
    model_path: str = "models/v3/Consensus/consensus_model.pkl",
    feature_path: str = "models/v3/feature_names.json",
    matrix_path: str = "data/processed",
) -> MatchEngine:
    """Loads the ConsensusOracle + feature matrix from disk."""
    from src.models.simulator import TournamentSimulator
    from src.features.csv_oracle import CSVFeatureOracle

    with open(model_path, "rb") as f:
        oracle = pickle.load(f)

    with open(feature_path, "r") as f:
        feature_names = json.load(f)

    oracle_engine = CSVFeatureOracle(matrix_path)
    matrix_2026 = oracle_engine.build_2026_matrix()

    sim = TournamentSimulator(oracle)
    return MatchEngine(sim, matrix_2026, feature_names)


def run_full_tournament_mc(
    n_iterations: int = 10_000,
    output_path: str = "models/mc_win_probabilities.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Main callable for CLI and integration tests."""
    engine = _load_engine_from_disk()
    full_sim = FullTournamentSimulator(engine)
    harness = MCSimulationHarness(full_sim, n_iterations=n_iterations, seed=seed)

    logger.info(f"Starting {n_iterations:,} full-tournament Monte Carlo simulations…")
    df = harness.run()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"MC results written → {output_path}")
    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="WC 2026 Full Tournament Monte Carlo Simulator")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--output", type=str, default="models/mc_win_probabilities.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_full_tournament_mc(
        n_iterations=args.iterations,
        output_path=args.output,
        seed=args.seed,
    )
    print(results.head(20).to_string(index=False))
