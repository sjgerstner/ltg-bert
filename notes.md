First, preprocess data:
```bash
./babylm/preprocessing/run.sh
```


```bash
python train.py --input_path ../data/processed_10M/all.txt --config_file ../configs/small.json --output_dir ../checkpoints/small --vocab_path ../tokenizer_10M.json
``` 

Conversion to huggingface:
```bash
python ltgbert_hf.py --model_pth ./babylm/checkpoints/ltgbert_base
```