conda init zsh
conda activate labmaite
python detect.py --weights ./custom/best-yolo-v1-3k.pt --img 640 --conf 0.4 --source "/Users/Dennis/Desktop/rpistage_test_data_210512_clean_chip/20x" --hide-labels