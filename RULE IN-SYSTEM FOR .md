Here is the rule in plain language, turned into something you can code against.

The game is a **stage-by-stage prediction contest**. Participants submit predictions **24 hours before each stage starts**, and once a stage prediction is submitted it **cannot be changed**. The stages are: **Group Stage, Round of 32, Round of 16, Quarter-final, Semi-final, and Final / 3rd Place**. The PDF also lists the scoring ladder by stage and says rankings are based on **total points**. 

### How the scoring works

For each match or fixture prediction, there are only three outcomes for scoring:

* **Correct scoreline** = you predicted the exact final score.
* **Correct result** = you got the winner/draw right, but not the exact score.
* **Incorrect** = both score and result are wrong.

The points increase as the tournament progresses:

* **Group Stage**: 100 for correct scoreline, 50 for correct result, 0 otherwise
* **Round of 32**: 100 / 50 / 0
* **Round of 16**: 150 / 75 / 0
* **Quarter-final**: 200 / 100 / 0
* **Semi-final**: 300 / 150 / 0
* **Final / 3rd Place**: 400 / 200 / 0 

### What this means for your prediction system

Your system should do four things:

1. **Lock predictions by stage**
   When a stage deadline passes, freeze all predictions for that stage.

2. **Score each fixture independently**
   For every predicted match, assign:

   * exact score = full stage points
   * right outcome only = half points
   * wrong = 0

3. **Sum total points across all stages**
   Overall leaderboard ranking is based on total accumulated points. The document also defines prizes for:
   **1st place, 2nd place, 3rd place, Group Stage Winner, Knockout Stage Winner, and Highest Correct Scoreline Predictor**. 

4. **Handle ties by pooling prize money**
   If two or more participants tie for a prize position, the prize money for those tied positions is combined. For example, a tie for 1st and 2nd means those two prize amounts are pooled; a tie for 1st among three people means 1st + 2nd + 3rd are pooled. 

### Important knockout rule

The PDF says that for knockout matches, **the final winning team prediction counts**. The text after that is truncated, so the document is clear on the winner part but not fully clear on whether scoreline should reflect regulation time only, extra time, or penalties. Your system should treat this as an **open rule detail to confirm** before finalizing scoring logic. 

### Best system design based on this rule

Use this structure:

* `stage`
* `match_id`
* `predicted_score_home`
* `predicted_score_away`
* `predicted_winner`
* `actual_score_home`
* `actual_score_away`
* `actual_winner`
* `points_awarded`
* `submitted_at`
* `locked_at`

Then score with logic like:

* if exact score matches → full points
* else if result matches → half points
* else → zero
