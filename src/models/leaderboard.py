from typing import List, Dict, Any

class LeaderboardSystem:
    """
    Manages the contest leaderboard and tie-breaking logic according to RULE-AND-SYSTEM.md.
    """
    
    def rank_participants(self, participant_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ranks participants based on:
        1. Total Points (descending)
        2. Most "Exact Score" predictions hit (descending)
        3. Most "Knockout Stage" points (descending)
        4. Timestamp of first prediction submitted (ascending - earliest wins)
        """
        
        # Sort using Python's stable sort with a tuple key
        # Since Python sorts ascending by default, we negate the descending fields.
        # For the datetime, earlier is smaller, so it sorts ascending correctly without negation.
        
        ranked = sorted(
            participant_stats,
            key=lambda p: (
                -p.get("total_points", 0),
                -p.get("exact_hits", 0),
                -p.get("ko_points", 0),
                p.get("first_pred_time")  # datetime object
            )
        )
        
        return ranked
