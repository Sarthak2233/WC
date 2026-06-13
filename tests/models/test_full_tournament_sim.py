"""
TDD tests for full_tournament_sim.py (WC 2026 – 12 groups of 4 teams).
All oracle calls are fully mocked – no disk I/O required.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.models.full_tournament_sim import (
    MatchEngine,
    FullTournamentSimulator,
    MCSimulationHarness,
    ThirdPlaceRankingEngine,
    FIFABracketAllocator,
    SimulationResult,
    TeamResult,
    _simulate_group,
    _run_knockout_round,
    _build_r32_bracket,
    WC2026_GROUPS,
    BEST3_SLOT_INDICES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mock_engine(goals_a: int = 2, goals_b: int = 1) -> MatchEngine:
    """MatchEngine whose simulate() always returns (goals_a, goals_b)."""
    mock_sim = MagicMock()
    mock_sim.simulate_match.return_value = (goals_a, goals_b)
    return MatchEngine(mock_sim, pd.DataFrame(), [])


# 4-team mini groups mirroring the real WC format
MINI_GROUPS = {
    "A": ["Brazil", "Argentina", "Germany", "France"],
    "B": ["Spain", "England", "Italy", "Portugal"],
}


def make_mini_simulator(goals_a: int = 2, goals_b: int = 1) -> FullTournamentSimulator:
    engine = make_mock_engine(goals_a, goals_b)
    return FullTournamentSimulator(engine, groups=MINI_GROUPS)


# ---------------------------------------------------------------------------
# TeamResult
# ---------------------------------------------------------------------------

class TestTeamResult:
    def test_gd_property(self):
        r = TeamResult(name="Brazil", pts=6, gf=5, ga=2)
        assert r.gd == 3

    def test_defaults(self):
        r = TeamResult(name="Phantasia")
        assert r.pts == 0
        assert r.stage == "group"

    def test_disciplinary_pts(self):
        r = TeamResult(name="X", yellow_cards=3, red_cards=1)
        assert r.disciplinary_pts == 6  # 3×1 + 1×3


# ---------------------------------------------------------------------------
# WC 2026 Format invariants
# ---------------------------------------------------------------------------

class TestWC2026Format:
    def test_twelve_groups(self):
        assert len(WC2026_GROUPS) == 12

    def test_four_teams_per_group(self):
        for grp, teams in WC2026_GROUPS.items():
            assert len(teams) == 4, f"Group {grp} has {len(teams)} teams, expected 4"

    def test_total_forty_eight_teams(self):
        all_teams = [t for teams in WC2026_GROUPS.values() for t in teams]
        assert len(all_teams) == 48

    def test_best3_slot_count(self):
        assert len(BEST3_SLOT_INDICES) == 8


# ---------------------------------------------------------------------------
# MatchEngine
# ---------------------------------------------------------------------------

class TestMatchEngine:
    def test_simulate_returns_tuple(self):
        engine = make_mock_engine(3, 0)
        ga, gb = engine.simulate("Brazil", "Argentina")
        assert ga == 3 and gb == 0

    def test_simulate_knockout_flag_forwarded(self):
        mock_sim = MagicMock()
        mock_sim.simulate_match.return_value = (1, 0)
        engine = MatchEngine(mock_sim, pd.DataFrame(), [])
        engine.simulate("A", "B", is_knockout=True)
        _, kwargs = mock_sim.simulate_match.call_args
        assert kwargs.get("is_knockout") is True

    def test_feature_lookup_hit(self):
        df = pd.DataFrame([{"canonical_team": "Brazil", "elo": 2100}])
        mock_sim = MagicMock()
        mock_sim.simulate_match.return_value = (2, 1)
        engine = MatchEngine(mock_sim, df, ["elo"])
        engine.simulate("Brazil", "Germany")
        team_a_feat = mock_sim.simulate_match.call_args[0][0]
        assert team_a_feat["elo"] == 2100

    def test_feature_lookup_miss_returns_dict(self):
        engine = make_mock_engine()
        feat = engine._get_team_features("NoSuchTeam")
        assert isinstance(feat, dict)


# ---------------------------------------------------------------------------
# Group Stage (4-team round-robin = 6 matches)
# ---------------------------------------------------------------------------

class TestSimulateGroup:
    def test_returns_four_results(self):
        engine = make_mock_engine(2, 0)
        results = _simulate_group("A", ["W", "X", "Y", "Z"], engine)
        assert len(results) == 4

    def test_all_are_team_results(self):
        engine = make_mock_engine(1, 0)
        results = _simulate_group("A", ["W", "X", "Y", "Z"], engine)
        assert all(isinstance(r, TeamResult) for r in results)

    def test_total_points_six_matches_all_decisive(self):
        """6 decisive matches → 6×3 = 18 pts total."""
        engine = make_mock_engine(1, 0)
        results = _simulate_group("A", ["W", "X", "Y", "Z"], engine)
        assert sum(r.pts for r in results) == 18

    def test_total_points_all_draws(self):
        """6 drawn matches → 6×2 = 12 pts total."""
        engine = make_mock_engine(1, 1)
        results = _simulate_group("A", ["W", "X", "Y", "Z"], engine)
        assert sum(r.pts for r in results) == 12

    def test_group_tag_assigned(self):
        engine = make_mock_engine(1, 0)
        results = _simulate_group("G", ["W", "X", "Y", "Z"], engine)
        assert all(r.group == "G" for r in results)


# ---------------------------------------------------------------------------
# ThirdPlaceRankingEngine
# ---------------------------------------------------------------------------

class TestThirdPlaceRankingEngine:
    def _make_thirds(self):
        return [
            TeamResult("A3", group="A", pts=4, gf=3, ga=1),
            TeamResult("B3", group="B", pts=4, gf=2, ga=1),
            TeamResult("C3", group="C", pts=3, gf=5, ga=3),
            TeamResult("D3", group="D", pts=1, gf=0, ga=0),
        ]

    def test_rank_returns_sorted(self):
        thirds = self._make_thirds()
        ranked = ThirdPlaceRankingEngine.rank(thirds)
        assert ranked[0].name == "A3"  # 4 pts, GD=+2

    def test_select_best_two(self):
        thirds = self._make_thirds()
        best = ThirdPlaceRankingEngine.select_best_n(thirds, n=2)
        assert len(best) == 2
        assert best[0].name == "A3"

    def test_disciplinary_tiebreak(self):
        """Between equal Pts/GD/GF, fewer cards wins."""
        t1 = TeamResult("Clean", group="E", pts=3, gf=1, ga=0, yellow_cards=0)
        t2 = TeamResult("Dirty", group="F", pts=3, gf=1, ga=0, yellow_cards=2)
        ranked = ThirdPlaceRankingEngine.rank([t2, t1])  # dirty first in input
        assert ranked[0].name == "Clean"


# ---------------------------------------------------------------------------
# FIFABracketAllocator
# ---------------------------------------------------------------------------

class TestFIFABracketAllocator:
    def _make_best_thirds(self):
        return [
            TeamResult(f"T{i}", group=chr(65 + i), pts=4) for i in range(8)
        ]

    def test_assigns_to_slots_24_31(self):
        bracket = [None] * 32
        best = self._make_best_thirds()
        result = FIFABracketAllocator.assign(best, bracket)
        filled = [result[i] for i in range(24, 32) if result[i] is not None]
        assert len(filled) == 8

    def test_no_overwrites_in_first_24(self):
        bracket = [f"Team{i}" for i in range(24)] + [None] * 8
        best = self._make_best_thirds()
        result = FIFABracketAllocator.assign(best, bracket)
        for i in range(24):
            assert result[i] == f"Team{i}", f"Slot {i} was overwritten"

    def test_no_duplicate_assignments(self):
        bracket = [None] * 32
        best = self._make_best_thirds()
        result = FIFABracketAllocator.assign(best, bracket)
        filled = [r for r in result if r is not None]
        assert len(filled) == len(set(filled))


# ---------------------------------------------------------------------------
# R32 Bracket Builder
# ---------------------------------------------------------------------------

class TestBuildR32Bracket:
    def _make_full_standings(self):
        winners, runners_up, thirds = {}, {}, []
        for grp, teams in WC2026_GROUPS.items():
            winners[grp] = TeamResult(name=teams[0], group=grp, pts=9)
            runners_up[grp] = TeamResult(name=teams[1], group=grp, pts=6)
            thirds.append(TeamResult(name=teams[2], group=grp, pts=3))
        return winners, runners_up, thirds

    def test_returns_32_teams(self):
        w, r, t = self._make_full_standings()
        bracket = _build_r32_bracket(w, r, t)
        assert len(bracket) == 32

    def test_no_duplicates(self):
        w, r, t = self._make_full_standings()
        bracket = _build_r32_bracket(w, r, t)
        real_teams = [x for x in bracket if x != "_bye_"]
        assert len(real_teams) == len(set(real_teams))

    def test_all_strings(self):
        w, r, t = self._make_full_standings()
        bracket = _build_r32_bracket(w, r, t)
        assert all(isinstance(s, str) for s in bracket)


# ---------------------------------------------------------------------------
# Knockout Round
# ---------------------------------------------------------------------------

class TestRunKnockoutRound:
    def test_halves_field(self):
        engine = make_mock_engine(2, 1)
        winners = _run_knockout_round(["A", "B", "C", "D"], engine, "R16")
        assert len(winners) == 2

    def test_bye_auto_advances(self):
        engine = make_mock_engine(0, 0)
        winners = _run_knockout_round(["_bye_", "Brazil"], engine, "R32")
        assert winners == ["Brazil"]

    def test_odd_teams_raises(self):
        engine = make_mock_engine()
        with pytest.raises(ValueError):
            _run_knockout_round(["A", "B", "C"], engine, "QF")


# ---------------------------------------------------------------------------
# FullTournamentSimulator
# ---------------------------------------------------------------------------

class TestFullTournamentSimulator:
    def test_run_returns_simulation_result(self):
        sim = make_mini_simulator()
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_champion_is_string(self):
        result = make_mini_simulator().run()
        assert isinstance(result.champion, str)

    def test_champion_from_known_teams(self):
        all_teams = {t for teams in MINI_GROUPS.values() for t in teams} | {"_bye_"}
        result = make_mini_simulator().run()
        assert result.champion in all_teams

    def test_semi_finalists_is_list(self):
        result = make_mini_simulator().run()
        assert isinstance(result.semi_finalists, list)

    def test_group_results_present(self):
        result = make_mini_simulator().run()
        assert set(result.group_results.keys()) == set(MINI_GROUPS.keys())


# ---------------------------------------------------------------------------
# MCSimulationHarness
# ---------------------------------------------------------------------------

class TestMCSimulationHarness:
    def _make_harness(self, n: int = 30) -> MCSimulationHarness:
        return MCSimulationHarness(make_mini_simulator(), n_iterations=n, seed=0)

    def test_returns_dataframe(self):
        df = self._make_harness(10).run()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = self._make_harness(10).run()
        required = {"team", "champion_prob", "finalist_prob",
                    "semifinalist_prob", "quarterfinalist_prob"}
        assert required.issubset(df.columns)

    def test_champion_probs_sum_to_one(self):
        df = self._make_harness(100).run()
        assert abs(df["champion_prob"].sum() - 1.0) < 0.01

    def test_sorted_descending(self):
        df = self._make_harness(50).run()
        probs = df["champion_prob"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_deterministic_with_seed(self):
        df1 = self._make_harness(30).run()
        df2 = self._make_harness(30).run()
        assert df1["champion_count"].tolist() == df2["champion_count"].tolist()

    def test_no_bye_in_results(self):
        df = self._make_harness(20).run()
        assert "_bye_" not in df["team"].values

    def test_all_counts_non_negative(self):
        df = self._make_harness(20).run()
        for col in ["champion_count", "finalist_count",
                    "semifinalist_count", "quarterfinalist_count"]:
            assert (df[col] >= 0).all()
