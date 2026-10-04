import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "issue_triage"))

from triage import (  # noqa: E402  # ty: ignore[unresolved-import]
    CATEGORIES,
    MARKER,
    Classification,
    CommentRenderer,
    IssueTriage,
)


class FakeGitHub:
    def __init__(self, issues):
        self._issues = issues
        self.comments = []
        self.labels = []

    def recent_issues(self, since):
        return self._issues

    def comment(self, number, body):
        self.comments.append((number, body))

    def add_labels(self, number, labels):
        self.labels.append((number, labels))


class FakeClassifier:
    def __init__(self, result):
        self._result = result

    def classify(self, title, body):
        return self._result


def _issue(number=1, labels=(), association="NONE"):
    return {
        "number": number,
        "title": "Model does not load",
        "body": "OSError: no such file",
        "labels": [{"name": name} for name in labels],
        "author_association": association,
        "user": {"login": "alice"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _triage(github, result, dry_run=False):
    return IssueTriage(github, FakeClassifier(result), lookback_minutes=35, min_confidence=0.8, dry_run=dry_run)


def test_model_download_template_renders_with_marker_and_author():
    text = CommentRenderer().render(CATEGORIES["model_download"], "alice")

    assert text.startswith(MARKER)
    assert "@alice" in text


def test_confident_model_download_issue_gets_comment_and_label():
    github = FakeGitHub([_issue()])

    _triage(github, Classification("model_download", 0.95, "r")).run()

    assert [n for n, _ in github.comments] == [1]
    assert github.labels == [(1, ["triage:model-download"])]


def test_low_confidence_and_other_categories_are_left_alone():
    github = FakeGitHub([_issue(1), _issue(2)])

    _triage(github, Classification("model_download", 0.5, "r")).run()
    _triage(github, Classification("other", 0.99, "r")).run()

    assert github.comments == []


def test_triaged_and_maintainer_issues_are_skipped():
    github = FakeGitHub([_issue(1, labels=["triage:model-download"]), _issue(2, association="MEMBER")])

    _triage(github, Classification("model_download", 0.99, "r")).run()

    assert github.comments == []


def test_dry_run_posts_nothing():
    github = FakeGitHub([_issue()])

    _triage(github, Classification("model_download", 0.99, "r"), dry_run=True).run()

    assert github.comments == [] and github.labels == []
