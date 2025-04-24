from huggingface_hub import snapshot_download
from jsonargparse import CLI


def download(fpath: str, base_model_repo_id: str, auth_token: str):
    snapshot_download(
        base_model_repo_id,
        local_dir=fpath,
        max_workers=4,
        token=auth_token
    )


if __name__ == "__main__":
    CLI(download)