CUDA_VISIBLE_DEVICES=3 python universal_main.py --data_path /home/xuemeng/reverb/datasets/mini_librispeech --epsilon 0.02  --alpha 0.0001 --num_iters 3000 --target_sentence " "

CUDA_VISIBLE_DEVICES=1 python main.py --input_wav /home/xuemeng/reverb/deepspeech2-pytorch-adversarial-attack/data/6078-54007-0001.wav --output_wav ./_.wav --epsilon 0.001  --alpha 0.0001 --num_iters 200 --target_sentence " "