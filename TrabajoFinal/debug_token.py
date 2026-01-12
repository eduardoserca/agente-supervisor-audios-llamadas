import config
import os

print(f"HUGGINGFACE_TOKEN from config: {'Presente' if config.HUGGINGFACE_TOKEN else 'AUSENTE'}")
print(f"HUGGINGFACE_TOKEN from env: {'Presente' if os.environ.get('HUGGINGFACE_TOKEN') else 'AUSENTE'}")
if config.HUGGINGFACE_TOKEN:
    print(f"Token length: {len(config.HUGGINGFACE_TOKEN)}")
