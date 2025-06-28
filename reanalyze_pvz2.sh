#!/bin/bash
# Script to re-analyze PvZ2 sessions with the fixed analyzer

echo "🔧 Re-analyzing PvZ2 sessions with fixed analyzer..."
echo "📝 This will use actual Gemini API calls to properly identify PvZ2 assets"
echo ""

# Analyze first session
echo "▶️  Analyzing session 1: 25-06-25_at_00.48.15"
python -m src.analysis.analyze data/PvZ2/25-06-25_at_00.48.15

echo ""
echo "▶️  Analyzing session 2: 25-06-25_at_00.42.21" 
python -m src.analysis.analyze data/PvZ2/25-06-25_at_00.42.21

echo ""
echo "✅ Re-analysis complete!"
echo "🎯 Now run the behavior tree generator to get proper PvZ2 behavior descriptions:"
echo "   python -m src.analysis.bt_generator --game PvZ2 --session 1 --format text" 