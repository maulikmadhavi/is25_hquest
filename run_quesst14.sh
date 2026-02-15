# Step1 - Extract features for audio, dev-query, and test-query sets
pixi run python ./src/feature_extraction.py --input_dir=data/quesst14Database/quesst14Database/Audio --output_csv=data/quesst14_audio.csv


pixi run python ./src/feature_extraction.py --input_dir=data/quesst14Database/quesst14Database/dev_queries --output_csv=data/quesst14_dev_queries.csv

pixi run python ./src/feature_extraction.py --input_dir=data/quesst14Database/quesst14Database/eval_queries --output_csv=data/quesst14_eval_queries.csv



# pixi run python main.py --audio_csv=data/quesst14_audio.csv --query_csv=data/quesst14_queries.csv --output_csv=data/quesst14_results.csv --top_k 10
