# Recording the `ptp_memory` demo

1. Install the project and run `python -m data.generate`.
2. Start a terminal recorder such as OBS Studio or Windows Game Bar.
3. Run `python -m app.cli --scenario ptp_memory`.
4. Capture both scripted calls, including the second call's memory-aware opener.
5. Trim the recording to the terminal interaction and export an MP4.
6. Keep the recording small enough for GitHub and do not capture `.env` or secrets.
