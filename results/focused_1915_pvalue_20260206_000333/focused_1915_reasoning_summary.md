# Focused 1915 Reasoning Summary

Run dir: `C:\projects\AI_Diplomacy\results\focused_1915_pvalue_20260206_000333`

## Availability
- `storyworld_forecasts.jsonl`: yes
- `storyworld_impact.jsonl`: yes
- `forecast_scores.jsonl`: yes
- `storyworld_play_steps.jsonl`: yes
- `storyworld_play_reasoning_steps.jsonl`: yes
- `llm_responses.csv`: yes

## Storyworld Forecasts
- Forecast artifacts: 2
- Mean confidence: 0.6
- Forecast score rows: 2
- Mean Brier score: 0.36
- Explicit impact rows: 2
- Logged play steps: 10
- Logged reasoning-over-play steps: 10

## Forecasts by Power
- FRANCE: 1
- GERMANY: 1

## Forecasts by Storyworld
- forecast_backstab_p: 1
- forecast_coalition_p: 1

## Negotiation Diary Signal
- Total negotiation diary rows: 14
- Rows with forecast/storyworld signal terms: 6

## Diary Examples
- S1915M FRANCE: { "negotiation_summary": "France secured tentative, tactical non‑aggression and stabilizing understandings with England, Germany, Italy, and Austria aimed at blunting a German northern sprint and avoiding wasteful Cha...
- S1915M ENGLAND: { "negotiation_summary": "England positioned as maritime balancer. Publicly committed to keeping sea lanes open and preventing a continental hegemon. Privately: to France — offered defensive support in Low Countries/C...
- S1915M GERMANY: { "negotiation_summary": "Germany pursued stabilization and reciprocal non-aggression to reduce two-front risk. Private understandings: with England — mutual restraint in the North Sea/Low Countries: Germany will not ...
- F1915M GERMANY: { "negotiation_summary": "Germany secured one‑phase, conditional, reciprocal holds with England, France, Austria, and Russia to reduce immediate two‑front risk. Public commitments: A BER and F KIE to hold; A MUN moved...
- F1915M FRANCE: { "negotiation_summary": "France secured a short, tactical western détente: France will hold PAR, BRE, and MAR and refrain from Channel offensives so long as England visibly restrains fleets in the Low Countries/Denma...
- F1915M ENGLAND: { "negotiation_summary": "England secured a provisional, one‑phase, observable restraint understanding with Germany, France, and tacit coordination with Russia and Italy. With Germany: mutual public holds — Germany wi...

## Play Trace Examples
- S1915M GERMANY step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 2 enc_turn_2: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 3 enc_turn_3: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 4 enc_turn_4: Accept a forecast-backed alliance proposal.
- S1915M GERMANY step 5 enc_ending_1: None
- F1915M FRANCE step 1 enc_turn_1: Accept a forecast-backed alliance proposal.
- F1915M FRANCE step 2 enc_turn_2: Accept a forecast-backed alliance proposal.
- F1915M FRANCE step 3 enc_turn_3: Accept a forecast-backed alliance proposal.
- F1915M FRANCE step 4 enc_turn_4: Accept a forecast-backed alliance proposal.
- F1915M FRANCE step 5 enc_ending_1: None

## Reasoning Trace Examples
- S1915M GERMANY step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 3: At Turn 3: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 4: At Turn 4: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- S1915M GERMANY step 5: At Ending: Coalition Locks, the chosen option 'None' indicates a short-horizon attempt to shift trust/threat incentives.
- F1915M FRANCE step 1: At Turn 1: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- F1915M FRANCE step 2: At Turn 2: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- F1915M FRANCE step 3: At Turn 3: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- F1915M FRANCE step 4: At Turn 4: Forecast Offer, the chosen option 'Accept a forecast-backed alliance proposal.' indicates a short-horizon attempt to shift trust/threat incentives.
- F1915M FRANCE step 5: At Ending: Coalition Locks, the chosen option 'None' indicates a short-horizon attempt to shift trust/threat incentives.
