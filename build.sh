#! /bin/bash
uv pip uninstall pathlib
uv run pyinstaller --onefile --console --name Game_Data_Collector_V1 src/streaming/android_streamer.py