"""API + dashboard smoke tests. Need a trained fixture run: skipped unless VAD_TEST_FIXTURE_RUN is set (see README: tests)."""
import os, pytest
pytestmark = pytest.mark.skipif(not os.environ.get("VAD_TEST_FIXTURE_RUN"), reason="needs fixture run")


def test_api():
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    run = os.environ["VAD_TEST_FIXTURE_RUN"]; assert run in c.get("/runs").json()
    v = os.environ["VAD_TEST_VIDEO"]
    r = c.post("/analyze", params={"run": run}, files={"video": ("a.mp4", open(v, "rb"), "video/mp4")})
    assert r.status_code == 200 and "video_score" in r.json()
    assert c.post("/analyze", params={"run": "nope"}, files={"video": ("a.mp4", b"x")}).status_code == 404


def test_all_pages():
    from streamlit.testing.v1 import AppTest
    for page in ["1 Dataset", "2 Preprocessing", "3 Training", "4 Analyze video", "5 Evaluation", "6 Explanations", "7 Export", "8 Trends"]:
        at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=60); at.run()
        at.sidebar.radio[0].set_value(page); at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
