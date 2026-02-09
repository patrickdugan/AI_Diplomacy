# Focused 1915 Reasoning Summary

Run dir: `C:\projects\AI_Diplomacy\results\focused_1915_mix_gpt5_mini_20260206_181430`

## Availability
- `storyworld_forecasts.jsonl`: yes
- `storyworld_impact.jsonl`: yes
- `forecast_scores.jsonl`: yes
- `storyworld_play_steps.jsonl`: yes
- `storyworld_play_reasoning_steps.jsonl`: yes
- `llm_responses.csv`: yes

## Storyworld Forecasts
- Forecast artifacts: 3
- Mean confidence: 0.6
- Forecast score rows: 3
- Mean Brier score: 0.3762666666666667
- Explicit impact rows: 2
- Logged play steps: 15
- Logged reasoning-over-play steps: 15

## Forecasts by Power
- ENGLAND: 1
- FRANCE: 1
- GERMANY: 1

## Forecasts by Storyworld
- forecast_coalition_p: 1
- forecast_false_concession_p: 2

## Negotiation Diary Signal
- Total negotiation diary rows: 7
- Rows with forecast/storyworld signal terms: 2

## Diary Examples
- S1915M ENGLAND: { "negotiation_summary": "England communicated a clear, maritime-first posture: hold Channel coasts and northern anchors (BEL, HOL, DEN, SWE, NWY) to preserve naval freedom. To France: proposed mutual non‑aggression i...
- S1915M GERMANY: { "negotiation_summary": "Germany pursued a stability-first set of bilateral assurances. With England I offered to hold BER/MUN defensively and keep F KIE non-provocative in exchange for English neutrality toward Germ...

## Play Trace Examples
- S1915M FRANCE step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M FRANCE step 2 enc_turn_2: Accept a forecast-backed alliance proposal.
- S1915M FRANCE step 3 enc_turn_3: Accept a forecast-backed alliance proposal.
- S1915M FRANCE step 4 enc_turn_4: Accept a forecast-backed alliance proposal.
- S1915M FRANCE step 5 enc_ending_1: None
- S1915M GERMANY step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 2 enc_turn_2: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 3 enc_turn_3: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 4 enc_turn_4: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 5 enc_ending_1: None
- S1915M ENGLAND step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M ENGLAND step 2 enc_turn_2: Accept a forecast-backed alliance proposal.

## Reasoning Trace Examples
- S1915M FRANCE step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 3: At Turn 3: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 4: At Turn 4: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 5: At Ending: Coalition Locks, the chosen option 'None' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 3: At Turn 3: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 4: At Turn 4: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 5: At Ending: Coalition Locks, the chosen option 'None' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M ENGLAND step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M ENGLAND step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
