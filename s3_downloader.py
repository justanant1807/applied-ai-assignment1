import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError

# --- Config you may change ---
BUCKET = "aft-vbi-pds"
IMG_PREFIX = "bin-images/"
META_PREFIX = "metadata/"
TARGET_PAIRS = 1000  # change as needed
REGION = "us-east-1"
# -----------------------------

def main():
    # unsigned (public) access, no AWS creds required
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name=REGION)

    os.makedirs("bin-images", exist_ok=True)
    os.makedirs("metadata", exist_ok=True)

    downloaded = 0
    token = None

    while downloaded < TARGET_PAIRS:
        kwargs = {"Bucket": BUCKET, "Prefix": IMG_PREFIX, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        # If bucket/prefix is empty
        contents = resp.get("Contents", [])
        if not contents:
            print("No objects found under prefix:", IMG_PREFIX)
            break

        for obj in contents:
            key = obj["Key"]
            fname = key.rsplit("/", 1)[-1]
            if not fname.endswith(".jpg"):
                continue

            # Download image
            local_img = os.path.join("bin-images", fname)
            s3.download_file(BUCKET, key, local_img)

            # Try matching metadata JSON
            meta_key = f"{META_PREFIX}{fname[:-4]}.json"
            local_meta = os.path.join("metadata", f"{fname[:-4]}.json")
            try:
                s3.download_file(BUCKET, meta_key, local_meta)
            except ClientError as e:
                # If you require strict pairs, uncomment to remove the image when JSON missing:
                # os.remove(local_img)
                # continue
                # Otherwise keep the image and don't count this as a pair
                continue

            downloaded += 1
            if downloaded % 50 == 0:
                print(f"Downloaded {downloaded}/{TARGET_PAIRS} pairs...")
            if downloaded >= TARGET_PAIRS:
                break

        if not resp.get("IsTruncated"):
            # No more pages
            break
        token = resp.get("NextContinuationToken")

    print(f"✅ Done: {downloaded} image+metadata pairs downloaded.")

if __name__ == "__main__":
    main()
