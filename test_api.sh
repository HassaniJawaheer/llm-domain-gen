set -a
source .env
set +a

curl http://localhost:$VLLM_PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "messages": [
      {
        "role": "user",
        "content": "FitFusion is a fitness studio that offers a unique blend of yoga, Pilates, and dance classes. Their expert instructors design fusion workouts that cater to different fitness levels, from beginners to advanced practitioners. Their studios are equipped with state-of-the-art equipment and provide a serene atmosphere for members to relax and rejuvenate."
      }
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
