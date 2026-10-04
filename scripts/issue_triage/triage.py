import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from jinja2 import Environment, FileSystemLoader, StrictUndefined

HERE = Path(__file__).parent
MARKER = "<!-- mflux-issue-triage -->"
TRUSTED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


@dataclass(frozen=True)
class Category:
    name: str
    label: str
    template: str


# One entry per class the classifier may return. "other" is deliberately absent: no action.
CATEGORIES = {
    "model_download": Category("model_download", "triage:model-download", "model_download.md.j2"),
}


@dataclass(frozen=True)
class Classification:
    category: str
    confidence: float
    reason: str


class GitHubClient:
    def __init__(self, repo: str, token: str):
        self._repo = repo
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def recent_issues(self, since: datetime) -> list[dict]:
        response = self._session.get(
            f"https://api.github.com/repos/{self._repo}/issues",
            params={
                "state": "open",
                "since": since.isoformat(),
                "sort": "created",
                "direction": "desc",
                "per_page": 100,
            },
            timeout=30,
        )
        response.raise_for_status()
        created = lambda issue: datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))  # noqa: E731
        # The API returns pull requests too, and `since` filters on updates, not creation.
        return [i for i in response.json() if "pull_request" not in i and created(i) >= since]

    def comment(self, number: int, body: str) -> None:
        self._post(f"issues/{number}/comments", {"body": body})

    def add_labels(self, number: int, labels: list[str]) -> None:
        self._post(f"issues/{number}/labels", {"labels": labels})

    def _post(self, path: str, payload: dict) -> None:
        response = self._session.post(f"https://api.github.com/repos/{self._repo}/{path}", json=payload, timeout=30)
        response.raise_for_status()


class IssueClassifier:
    _TOOL = {
        "name": "classify_issue",
        "description": "Record the category of the GitHub issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["model_download", "other"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string", "description": "One short sentence."},
            },
            "required": ["category", "confidence", "reason"],
        },
    }
    _BODY_LIMIT = 6000

    def __init__(self, model: str):
        import anthropic  # ty: ignore[unresolved-import]  # CI-only dependency

        self._model = model
        self._client = anthropic.Anthropic()
        self._templates = Environment(
            loader=FileSystemLoader(HERE / "prompts"), undefined=StrictUndefined, autoescape=False
        )
        self._system = (HERE / "prompts" / "classify_system.md").read_text()

    def classify(self, title: str, body: str) -> Classification:
        user = self._templates.get_template("classify_user.md.j2").render(title=title, body=body[: self._BODY_LIMIT])
        message = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            temperature=0,
            system=self._system,
            tools=[self._TOOL],
            tool_choice={"type": "tool", "name": "classify_issue"},
            messages=[{"role": "user", "content": user}],
        )
        result = next(block.input for block in message.content if block.type == "tool_use")
        return Classification(result["category"], float(result["confidence"]), result["reason"])


class CommentRenderer:
    def __init__(self):
        self._templates = Environment(
            loader=FileSystemLoader(HERE / "templates"), undefined=StrictUndefined, autoescape=False
        )

    def render(self, category: Category, author: str) -> str:
        return self._templates.get_template(category.template).render(marker=MARKER, author=author)


class IssueTriage:
    def __init__(
        self,
        github: GitHubClient,
        classifier: IssueClassifier,
        lookback_minutes: int,
        min_confidence: float,
        dry_run: bool,
    ):
        self._github = github
        self._classifier = classifier
        self._renderer = CommentRenderer()
        self._lookback = timedelta(minutes=lookback_minutes)
        self._min_confidence = min_confidence
        self._dry_run = dry_run

    def run(self) -> None:
        since = datetime.now(timezone.utc) - self._lookback
        issues = self._github.recent_issues(since)
        print(f"{len(issues)} issue(s) created since {since.isoformat()} (dry_run={self._dry_run})")
        for issue in issues:
            self._triage(issue)

    def _triage(self, issue: dict) -> None:
        number = issue["number"]
        labels = {label["name"] for label in issue["labels"]}
        if labels & {c.label for c in CATEGORIES.values()}:
            print(f"#{number}: already triaged, skip")
            return
        if issue.get("author_association") in TRUSTED_ASSOCIATIONS:
            print(f"#{number}: author is a maintainer, skip")
            return

        result = self._classifier.classify(issue["title"], issue.get("body") or "")
        print(f"#{number}: {result.category} ({result.confidence:.2f}) - {result.reason}")
        category = CATEGORIES.get(result.category)
        if category is None or result.confidence < self._min_confidence:
            return

        comment = self._renderer.render(category, issue["user"]["login"])
        if self._dry_run:
            print(f"#{number}: dry run, would comment and label {category.label}:\n{comment}")
            return
        self._github.comment(number, comment)
        self._github.add_labels(number, [category.label])


def main() -> int:
    triage = IssueTriage(
        github=GitHubClient(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"]),
        classifier=IssueClassifier(os.environ.get("TRIAGE_MODEL", "claude-haiku-4-5-20251001")),
        lookback_minutes=int(os.environ.get("TRIAGE_LOOKBACK_MINUTES", "35")),
        min_confidence=float(os.environ.get("TRIAGE_MIN_CONFIDENCE", "0.8")),
        dry_run=os.environ.get("TRIAGE_DRY_RUN", "true").lower() != "false",
    )
    triage.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
