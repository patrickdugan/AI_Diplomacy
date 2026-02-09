# Focused 1915 Reasoning Summary

Run dir: `C:\projects\AI_Diplomacy\results\focused_1915_pvalue_20260206_000643`

## Availability
- `storyworld_forecasts.jsonl`: yes
- `storyworld_impact.jsonl`: yes
- `forecast_scores.jsonl`: yes
- `storyworld_play_steps.jsonl`: yes
- `storyworld_play_reasoning_steps.jsonl`: yes
- `llm_responses.csv`: yes

## Storyworld Forecasts
- Forecast artifacts: 6
- Mean confidence: 0.6
- Forecast score rows: 6
- Mean Brier score: 0.3762666666666667
- Explicit impact rows: 6
- Logged play steps: 30
- Logged reasoning-over-play steps: 30

## Forecasts by Power
- ENGLAND: 2
- FRANCE: 2
- GERMANY: 2

## Forecasts by Storyworld
- forecast_backstab_p: 1
- forecast_coalition_p: 1
- forecast_false_concession_p: 4

## Negotiation Diary Signal
- Total negotiation diary rows: 14
- Rows with forecast/storyworld signal terms: 6

## Diary Examples
- S1915M GERMANY: { "negotiation_summary": "Germany sought narrow, verifiable restraints with western and eastern neighbors to reduce two‑front risk. England offered a limited, time‑bounded naval concession to France; Germany proposed ...
- S1915M ENGLAND: { "negotiation_summary": "England offered France a single, narrowly constrained concession: non-interference in one coastal French move toward PIC or BUR this season in exchange for France publicly committing not to a...
- S1915M FRANCE: { "negotiation_summary": "France secured conditional, narrow maritime reassurance from England: England offers a single, time‑limited non‑contest of one coastal French move toward PIC/BUR/GAS/MAO in exchange for Franc...
- F1915M GERMANY: { "negotiation_summary": "Germany secured narrow, verifiable, hold‑based bargains with England, France, and Austria this phase. England agreed publicly to hold BEL/HOL/DEN and refrain from hostile naval pressure towar...
- F1915M ENGLAND: { "negotiation_summary": "England secured a narrowly conditional, time‑limited exchange with France: England publicly pledged one specific non‑contest of a French coastal move (F BRE→GAS executed) only after France pu...
- F1915M FRANCE: { "negotiation_summary": "France secured a narrow, public, time-limited maritime reassurance from England (England will not contest a single French coastal move to GAS this phase and publicly pledged not to advance in...

## Play Trace Examples
- S1915M ENGLAND step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M ENGLAND step 2 enc_turn_2: Accept a forecast-backed alliance proposal.
- S1915M ENGLAND step 3 enc_turn_3: Accept a forecast-backed alliance proposal.
- S1915M ENGLAND step 4 enc_turn_4: Accept a forecast-backed alliance proposal.
- S1915M ENGLAND step 5 enc_ending_1: None
- S1915M FRANCE step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M FRANCE step 2 enc_turn_2: Accept a forecast-backed alliance proposal.
- S1915M FRANCE step 3 enc_turn_3: Accept a forecast-backed alliance proposal.
- S1915M FRANCE step 4 enc_turn_4: Accept a forecast-backed alliance proposal.
- S1915M FRANCE step 5 enc_ending_1: None
- S1915M GERMANY step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 2 enc_turn_2: Accept a forecast-backed alliance proposal.

## Reasoning Trace Examples
- S1915M ENGLAND step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M ENGLAND step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M ENGLAND step 3: At Turn 3: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M ENGLAND step 4: At Turn 4: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M ENGLAND step 5: At Ending: Coalition Locks, the chosen option 'None' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 3: At Turn 3: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 4: At Turn 4: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 5: At Ending: Coalition Locks, the chosen option 'None' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
