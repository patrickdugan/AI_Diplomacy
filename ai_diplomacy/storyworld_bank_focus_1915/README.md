Focused 1915 Storyworld Bank

- `*.json` at this folder root are lightweight forecast templates used by the AI Diplomacy harness.
- `full_storyworlds/*.json` are full SweepWeave storyworlds for manual QA in the editor.

To refresh both sets:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\focused_1915\prepare_bank.ps1
```

Template files stay deterministic for runs, while `full_storyworlds` gives complete inspectable content.
