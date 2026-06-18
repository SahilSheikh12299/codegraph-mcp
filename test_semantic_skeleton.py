"""Unit tests for AST semantic skeleton extraction used in cross-encoder reranking."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))

from fileParsing.scanAST import build_semantic_skeleton
from advanced_engine import build_rerank_passage

SAMPLE_FUNCTION = '''
def resolve_redirects(self, resp, req, stream=False, timeout=None, verify=True, cert=None, proxies=None, yield_requests=False, **adapter_kwargs):
    """Follow HTTP redirects and return the final response."""
    redirect_location = resp.headers.get("Location")
    for r in resp.history:
        if not r.is_redirect:
            continue
        prepared_request = self.prepare_redirect(resp, req)
        resp = self.send(prepared_request, **adapter_kwargs)
    return resp
'''

SAMPLE_CLASS = '''
class SessionRedirectMixin:
    """Mixin for redirect handling."""

    MAX_REDIRECTS = 30

    def resolve_redirects(self, resp, req):
        """Resolve redirect chain."""
        return resp

    def get_redirect_target(self, resp):
        return resp.headers.get("location")
'''


def test_function_skeleton_excludes_control_flow():
    skeleton = build_semantic_skeleton(SAMPLE_FUNCTION, "FUNCTION")
    assert "resolve_redirects" in skeleton
    assert "Follow HTTP redirects" in skeleton
    assert "Calls:" in skeleton
    assert "prepare_redirect" in skeleton or "send" in skeleton
    assert "Variables:" in skeleton
    assert "redirect_location" in skeleton
    assert "for r in resp.history" not in skeleton
    assert "continue" not in skeleton


def test_class_skeleton_includes_methods_not_bodies():
    skeleton = build_semantic_skeleton(SAMPLE_CLASS, "CLASS")
    assert "SessionRedirectMixin" in skeleton
    assert "MAX_REDIRECTS" in skeleton or "Attributes:" in skeleton
    assert "resolve_redirects" in skeleton
    assert "get_redirect_target" in skeleton
    assert "for r in" not in skeleton


def test_build_rerank_passage_uses_skeleton():
    data = {
        "type": "FUNCTION",
        "name": "resolve_redirects",
        "chunk_text": SAMPLE_FUNCTION,
        "embedding_text": "Type: FUNCTION\nName: resolve_redirects",
        "belongs_to_class": "SessionRedirectMixin",
        "file_path": "sessions.py",
    }
    passage = build_rerank_passage(data)
    assert "Type: FUNCTION" in passage
    assert "Calls:" in passage
    assert "for r in resp.history" not in passage


if __name__ == "__main__":
    tests = [
        test_function_skeleton_excludes_control_flow,
        test_class_skeleton_includes_methods_not_bodies,
        test_build_rerank_passage_uses_skeleton,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {test.__name__} — {e}")
    sys.exit(1 if failed else 0)
