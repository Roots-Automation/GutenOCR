uv pip install "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run hf download "rootsautomation/TABMEpp" --repo-type dataset --local-dir .