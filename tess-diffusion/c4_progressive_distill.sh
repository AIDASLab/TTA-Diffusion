python run_train_c4.py \
    --model_name_or_path '' \
    --output_dir 'tess-diffusion/${LOCAL_DIR}/outputs/paper_experiments/c4_progressive/1000' \
    --num_diffusion_steps 1000 \
    --num_inference_diffusion_steps 1000 \
    --learning_rate 5e-6 \
    --max_steps 300000 \
    --logging_steps 1000 \
    --save_steps 20000 \
    --eval_steps 10000

python run_train_c4.py \
    --model_name_or_path '' \
    --output_dir 'tess-diffusion/${LOCAL_DIR}/outputs/paper_experiments/c4_progressive/200' \
    --num_diffusion_steps 200 \
    --num_inference_diffusion_steps 200 \
    --learning_rate 5e-6 \
    --max_steps 100000 \
    --logging_steps 1000 \
    --save_steps 20000 \
    --eval_steps 10000

python run_train_c4.py \
    --model_name_or_path '' \
    --output_dir 'tess-diffusion/${LOCAL_DIR}/outputs/paper_experiments/c4_progressive/100' \
    --num_diffusion_steps 100 \
    --num_inference_diffusion_steps 100 \
    --learning_rate 5e-6 \
    --max_steps 100000 \
    --logging_steps 1000 \
    --save_steps 20000 \
    --eval_steps 10000

python run_train_c4.py \
    --model_name_or_path '' \
    --output_dir 'tess-diffusion/${LOCAL_DIR}/outputs/paper_experiments/c4_progressive/50' \
    --num_diffusion_steps 50 \
    --num_inference_diffusion_steps 50 \
    --learning_rate 5e-6 \
    --max_steps 40000 \
    --logging_steps 1000 \
    --save_steps 20000 \
    --eval_steps 10000
