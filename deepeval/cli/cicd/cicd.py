import os
import json
import requests

def _post_github_pr_comment(markdown_content: str):
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")  # format: "owner/repo"
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    if not token or not repo or not event_path:
        missing = [
            name
            for name, val in (
                ("GITHUB_TOKEN", token),
                ("GITHUB_REPOSITORY", repo),
                ("GITHUB_EVENT_PATH", event_path),
            )
            if not val
        ]
        print(
            "Skipping PR comment: missing environment variable(s): "
            + ", ".join(missing)
            + ". (GitHub Actions sets these automatically; local runs skip posting.)"
        )
        return

    # Extract the PR number from the event payload
    try:
        with open(event_path, "r") as f:
            event_data = json.load(f)
            if "pull_request" not in event_data:
                print("Event is not a pull request. Skipping PR comment.")
                return
            pr_number = event_data["pull_request"]["number"]
    except Exception as e:
        print(f"Failed to read GitHub event payload: {e}")
        return

    base_url = (
        f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    )
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        # 1. Fetch existing comments
        response = requests.get(base_url, headers=headers)
        if response.status_code != 200:
            print(
                f"⚠️ Failed to fetch existing comments from GitHub. Status: {response.status_code}, Response: {response.text}"
            )
            return

        existing_comments = response.json()

        if not isinstance(existing_comments, list):
            print(
                f"⚠️ Unexpected response format from GitHub API when fetching comments. Expected a list, got: {type(existing_comments).__name__}. Response: {existing_comments}"
            )
            return

        # 2. Look for our specific bot comment
        comment_id_to_update = None
        for comment in existing_comments:
            if not isinstance(comment, dict):
                continue
            if "🚀 DeepEval Evaluation Results" in comment.get("body", ""):
                comment_id_to_update = comment["id"]
                break

        # 3. Update or Create
        if comment_id_to_update:
            patch_url = f"https://api.github.com/repos/{repo}/issues/comments/{comment_id_to_update}"
            response = requests.patch(
                patch_url, json={"body": markdown_content}, headers=headers
            )
            action = "updated"
        else:
            response = requests.post(
                base_url, json={"body": markdown_content}, headers=headers
            )
            action = "posted"

        if response.status_code in [200, 201]:
            print(f"✅ Successfully {action} DeepEval results on GitHub PR.")
        else:
            print(
                f"⚠️ Failed to {action} PR comment. Status: {response.status_code}, Response: {response.text}"
            )

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Network error while posting to GitHub: {e}")
