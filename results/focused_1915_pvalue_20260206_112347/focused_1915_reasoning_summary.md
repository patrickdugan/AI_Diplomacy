# Focused 1915 Reasoning Summary

Run dir: `C:\projects\AI_Diplomacy\results\focused_1915_pvalue_20260206_112347`

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
- Mean Brier score: 0.2962666666666667
- Explicit impact rows: 6
- Logged play steps: 30
- Logged reasoning-over-play steps: 27

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
- S1915M ENGLAND: { "negotiation_summary": "West: I offered France a limited armistice (no hit on Brest if they avoid Channel/Low Countries). France countered: I vacate SPA to MAO so they take MAR–SPA; they’d avoid ENG and use BRE–MAO ...
- S1915M GERMANY: { "negotiation_summary": "Established restrained, security‑focused understandings with England, France, Austria, Russia, and Turkey. England: maritime understanding — F KIE will hold/avoid DEN/HOL if England refrains ...
- S1915M FRANCE: { "negotiation_summary": "England offered a limited western armistice and not to hit Brest if I avoid ENG/Low Countries pushes; he flagged a 62% risk I would over‑commit after a concession and will keep cover. I count...
- F1915M ENGLAND: { "negotiation_summary": "Status quo secured in the North/Low Countries and at Kiel. Germany confirmed F KIE H; no English supports vs KIE/BER this phase, and public predictability requested. Russia affirmed quiet in ...
- F1915M GERMANY: { "negotiation_summary": "Germany secured conditional, verifiable holds with England, France, Austria, Russia, and Turkey. Commitments: F KIE — HOLD (publicly signalled to England); A BER — HOLD; A MUN — HOLD/READY to...
- F1915M FRANCE: { "negotiation_summary": "England refused to vacate Spain and insists on Iberian status quo with Channel calm; he cites a 62% artifact risk of French overreach and threatens coalition if I lunge at POR or ENG. I reite...

## Play Trace Examples
- S1915M GERMANY step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 2 enc_turn_2: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 3 enc_turn_3: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 4 enc_turn_4: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 5 enc_ending_1: None
- S1915M ENGLAND step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M ENGLAND step 2 enc_turn_2: Accept a forecast-backed alliance proposal.
- S1915M ENGLAND step 3 enc_turn_3: Accept a forecast-backed alliance proposal.
- S1915M ENGLAND step 4 enc_turn_4: Accept a forecast-backed alliance proposal.
- S1915M ENGLAND step 5 enc_ending_1: None
- S1915M FRANCE step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M FRANCE step 2 enc_turn_2: Accept a forecast-backed alliance proposal.

## Reasoning Trace Examples
- S1915M GERMANY step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 3: At Turn 3: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 4: At Turn 4: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 5: At Ending: Coalition Locks, the chosen option 'None' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M ENGLAND step 1: By accepting a forecast-backed alliance immediately, we signal preference for a stabilizing coalition anchored on credible naval-risk projections. This estab...
- S1915M ENGLAND step 2: A second acceptance reinforces a pattern: we reward partners who ground proposals in forecasted fleet movements. This nudges counterparts to share sailing ti...
- S1915M ENGLAND step 3: A third acceptance converts bilateral trust into a nascent multilateral frame. We leverage forecast consensus to prevent any single ally from overconcentrati...
- S1915M ENGLAND step 4: The fourth acceptance consolidates a coalition whose cohesion is underwritten by shared forecasts. We trade forecast clarity for compliance: we keep sea lane...
- S1915M FRANCE step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M FRANCE step 3: At Turn 3: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
