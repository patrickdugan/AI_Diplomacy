 = 'C:\projects\GPTStoryworld'
 = 'C:\projects\AI_Diplomacy\ai_diplomacy\storyworld_bank_focus_1915\templates'
 = '1'
 = '1'
 = 'model'
 = '8'
 = '1'
 = 'ENGLAND:forecast_false_concession_p'
 = '1'
 =  results\focused_1915_mix_gpt5_mini_20260206_181139
python lm_game.py --run_dir  --forecasting_analysis_mode true --forecasting_focus_powers ENGLAND,FRANCE,GERMANY --forecasting_state_file ai_diplomacy\scenarios\forecasting_1915_press.json --prompts_dir ai_diplomacy\prompts_forecasting --models gpt-5,gpt-5-mini,gpt-5,gpt-5-mini,gpt-5-mini,gpt-5,gpt-5-mini --max_year 1915 --end_at_phase W1915A --num_negotiation_rounds 1 --max_tokens 700 --simple_prompts false --generate_phase_summaries false
