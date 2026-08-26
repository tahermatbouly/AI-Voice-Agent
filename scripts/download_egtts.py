from huggingface_hub import snapshot_download

REPO = "OmarSamir/EGTTS-V0.1"

print("=" * 60)
print("Downloading EGTTS-V0.1")
print("=" * 60)
print(f"Repository: {REPO}")
print()

path = snapshot_download(
    repo_id=REPO,
    repo_type="model",
)
    
print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
print(f"Model cache: {path}")
