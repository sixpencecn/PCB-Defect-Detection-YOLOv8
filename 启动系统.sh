#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
unset QT_QPA_PLATFORM_PLUGIN_PATH QT_PLUGIN_PATH QT_QPA_PLATFORM
python main.py
