#!/bin/bash

# input_dir=./babylm/checkpoints/ltgbert_small_elc_tok
# tokenizer_pth=./babylm/tokenizer_10M_elcbert.json
# output_pth=/scratch/myh2014/evaluation-pipeline/models/ltgbert_10M

# for i in 0 10 20
# do
#     for epoch in 200 400 600 800
#     do 
#         python ltgbert_hf.py --model_pth "$input_dir/model_${epoch}_${i}.bin" --tokenizer_pth "$tokenizer_pth" --output_pth "${output_pth}-${epoch}-${i}" --is_small True
#     done
# done

input_dir=./babylm/checkpoints/ltgbert_base_v2_3e3
tokenizer_pth=./babylm/tokenizer.json
output_pth=/scratch/myh2014/evaluation-pipeline/models/

for i in 0 10 20
do
    for epoch in 1 3 7 11 15 19
    do 
        python ltgbert_hf.py --model_pth "$input_dir/model_${epoch}_${i}.bin" --tokenizer_pth "$tokenizer_pth" --output_pth "${output_pth}ltgbert_base/-${epoch}-${i}" --is_small False
    done
done

input_dir=./babylm/checkpoints/elcbert_base_3e3
for i in 20
do
    for epoch in 19
    do
        python elcbert_hf.py --model_pth "$input_dir/model_${epoch}_${i}.bin" --tokenizer_pth "$tokenizer_pth" --output_pth "${output_pth}elcbert_base/-${epoch}-${i}" --is_small False
    done
done

