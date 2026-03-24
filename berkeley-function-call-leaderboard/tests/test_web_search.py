from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import web_search
from requests import HTTPError, Response


class FakeResponse:
    def __init__(self, status_code=200, json_payload=None, text="", raise_error=None):
        self.status_code = status_code
        self._json_payload = json_payload
        self.text = text
        self._raise_error = raise_error

    def raise_for_status(self):
        if self._raise_error is not None:
            raise self._raise_error

    def json(self):
        if self._json_payload is None:
            raise ValueError("No JSON payload configured")
        return self._json_payload


def test_search_engine_query_prefers_serpapi(monkeypatch, capsys):
    api = web_search.WebSearchAPI()
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    class FakeGoogleSearch:
        def __init__(self, params):
            self.params = params

        def get_dict(self):
            return {
                "organic_results": [
                    {
                        "title": "Result from SerpAPI",
                        "link": "https://example.com/serp",
                        "snippet": "Serp snippet",
                    }
                ]
            }

    def fail_requests_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called when SerpAPI succeeds")

    monkeypatch.setattr(web_search, "GoogleSearch", FakeGoogleSearch)
    monkeypatch.setattr(web_search.requests, "get", fail_requests_get)

    results = api.search_engine_query("query", max_results=1)

    assert results == [
        {
            "title": "Result from SerpAPI",
            "href": "https://example.com/serp",
            "body": "Serp snippet",
        }
    ]
    assert "Provider 'serpapi' returned 1 results." in capsys.readouterr().out


def test_search_engine_query_falls_back_to_brave(monkeypatch, capsys):
    api = web_search.WebSearchAPI()
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")

    class FakeGoogleSearch:
        def __init__(self, params):
            self.params = params

        def get_dict(self):
            return {"error": "downstream failure"}

    def fake_requests_get(url, headers=None, params=None, timeout=None):
        assert url == "https://api.search.brave.com/res/v1/web/search"
        assert headers["X-Subscription-Token"] == "brave-key"
        return FakeResponse(
            json_payload={
                "web": {
                    "results": [
                        {
                            "title": "Result from Brave",
                            "url": "https://example.com/brave",
                            "description": "Brave snippet",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(web_search, "GoogleSearch", FakeGoogleSearch)
    monkeypatch.setattr(web_search.requests, "get", fake_requests_get)

    results = api.search_engine_query("query", max_results=1)

    assert results == [
        {
            "title": "Result from Brave",
            "href": "https://example.com/brave",
            "body": "Brave snippet",
        }
    ]
    output = capsys.readouterr().out
    assert "Provider 'serpapi' error: downstream failure" in output
    assert "Provider 'brave' returned 1 results." in output


def test_search_engine_query_falls_back_to_ddg_html(monkeypatch, capsys):
    api = web_search.WebSearchAPI()
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    html = """
    <html>
      <body>
        <div class="result">
          <a class="result__a" href="https://example.com/ddg">Result from DDG</a>
          <a class="result__snippet">DDG snippet</a>
        </div>
      </body>
    </html>
    """

    def fake_requests_get(url, headers=None, params=None, timeout=None):
        assert url == "https://html.duckduckgo.com/html/"
        return FakeResponse(text=html)

    monkeypatch.setattr(web_search.requests, "get", fake_requests_get)

    results = api.search_engine_query("query", max_results=1)

    assert results == [
        {
            "title": "Result from DDG",
            "href": "https://example.com/ddg",
            "body": "DDG snippet",
        }
    ]
    output = capsys.readouterr().out
    assert "Provider 'serpapi' skip: missing SERPAPI_API_KEY" in output
    assert "Provider 'brave' skip: missing BRAVE_SEARCH_API_KEY" in output
    assert "Provider 'duckduckgo-html' returned 1 results." in output


def test_search_engine_query_omits_snippets_when_disabled(monkeypatch):
    api = web_search.WebSearchAPI()
    api.show_snippet = False
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    html = """
    <html>
      <body>
        <div class="result">
          <a class="result__a" href="https://example.com/ddg">Result from DDG</a>
          <div class="result__snippet">DDG snippet</div>
        </div>
      </body>
    </html>
    """

    monkeypatch.setattr(
        web_search.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(text=html),
    )

    results = api.search_engine_query("query", max_results=1)

    assert results == [
        {
            "title": "Result from DDG",
            "href": "https://example.com/ddg",
        }
    ]


def test_search_engine_query_reports_exhausted_providers(monkeypatch, capsys):
    api = web_search.WebSearchAPI()
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")

    class FakeGoogleSearch:
        def __init__(self, params):
            self.params = params

        def get_dict(self):
            return {"error": "serp unavailable"}

    def fake_requests_get(url, headers=None, params=None, timeout=None):
        if "brave" in url:
            return FakeResponse(status_code=503, text="brave unavailable")
        return FakeResponse(
            text="<html><body><div class='not-a-result'>empty</div></body></html>"
        )

    monkeypatch.setattr(web_search, "GoogleSearch", FakeGoogleSearch)
    monkeypatch.setattr(web_search.requests, "get", fake_requests_get)
    monkeypatch.setattr(web_search.time, "sleep", lambda _: None)

    results = api.search_engine_query("query", max_results=1)

    assert results == {
        "error": (
            "Failed to retrieve web search results after trying providers "
            "serpapi, brave, duckduckgo-html."
        )
    }
    output = capsys.readouterr().out
    assert "Provider 'serpapi' error: serp unavailable" in output
    assert "Provider 'brave' error:" in output
    assert "Provider 'duckduckgo-html' error: no parsable HTML results returned" in output


def test_fetch_url_content_returns_direct_raw(monkeypatch, capsys):
    api = web_search.WebSearchAPI()

    def fake_get(url, headers=None, timeout=None, allow_redirects=None):
        assert url == "https://example.com/page"
        return FakeResponse(text="<html><body>Hello</body></html>")

    monkeypatch.setattr(web_search.requests, "get", fake_get)

    result = api.fetch_url_content("https://example.com/page", mode="raw")

    assert result == {"content": "<html><body>Hello</body></html>"}
    assert "Fetch backend 'direct' succeeded." in capsys.readouterr().out


def test_fetch_url_content_falls_back_to_cloudflare_on_timeout(monkeypatch, capsys):
    api = web_search.WebSearchAPI()
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "cf-account")

    def fake_get(*args, **kwargs):
        raise web_search.requests.Timeout("direct timed out")

    def fake_post(url, headers=None, json=None, timeout=None):
        assert url.endswith("/accounts/cf-account/browser-rendering/content")
        assert headers["Authorization"] == "Bearer cf-token"
        assert json == {"url": "https://example.com/page"}
        return FakeResponse(json_payload={"result": "<html><body>Cloudflare</body></html>"})

    monkeypatch.setattr(web_search.requests, "get", fake_get)
    monkeypatch.setattr(web_search.requests, "post", fake_post)

    result = api.fetch_url_content("https://example.com/page", mode="raw")

    assert result == {"content": "<html><body>Cloudflare</body></html>"}
    output = capsys.readouterr().out
    assert "Fetch backend 'direct' error: direct timed out" in output
    assert "Fetch backend 'cloudflare-browser-rendering' succeeded." in output


def test_fetch_url_content_falls_back_to_cloudflare_on_challenge_page(
    monkeypatch, capsys
):
    api = web_search.WebSearchAPI()
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "cf-account")

    challenge_html = "<html><body>Checking your browser before accessing</body></html>"

    monkeypatch.setattr(
        web_search.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(text=challenge_html),
    )
    monkeypatch.setattr(
        web_search.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(json_payload={"result": "# Cloudflare markdown"}),
    )

    result = api.fetch_url_content("https://example.com/page", mode="markdown")

    assert result == {"content": "# Cloudflare markdown"}
    output = capsys.readouterr().out
    assert "challenge page detected in direct fetch response" in output
    assert "Fetch backend 'cloudflare-browser-rendering' succeeded." in output


def test_fetch_url_content_direct_403_triggers_cloudflare(monkeypatch):
    api = web_search.WebSearchAPI()
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "cf-account")

    response = Response()
    response.status_code = 403
    response._content = b"forbidden"

    def fake_get(*args, **kwargs):
        raise HTTPError("403 Client Error: Forbidden", response=response)

    monkeypatch.setattr(web_search.requests, "get", fake_get)
    monkeypatch.setattr(
        web_search.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            json_payload={"result": "<html><body>Cloudflare</body></html>"}
        ),
    )

    result = api.fetch_url_content("https://example.com/page", mode="truncate")

    assert result == {"content": "Cloudflare"}


def test_fetch_url_content_reports_backend_chain_failure(monkeypatch, capsys):
    api = web_search.WebSearchAPI()
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "cf-account")

    def fake_get(*args, **kwargs):
        raise web_search.requests.Timeout("direct timed out")

    def fake_post(*args, **kwargs):
        raise web_search.requests.Timeout("cloudflare timed out")

    monkeypatch.setattr(web_search.requests, "get", fake_get)
    monkeypatch.setattr(web_search.requests, "post", fake_post)

    result = api.fetch_url_content("https://example.com/page", mode="raw")

    assert result == {
        "error": (
            "An error occurred while fetching https://example.com/page after trying backends "
            "direct, cloudflare-browser-rendering: cloudflare timed out"
        )
    }
    output = capsys.readouterr().out
    assert "Attempting fetch backend 'cloudflare-browser-rendering'" in output


def test_fetch_url_content_uses_direct_error_when_cloudflare_not_configured(monkeypatch):
    api = web_search.WebSearchAPI()
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)

    def fake_get(*args, **kwargs):
        raise web_search.requests.Timeout("direct timed out")

    monkeypatch.setattr(web_search.requests, "get", fake_get)

    result = api.fetch_url_content("https://example.com/page", mode="raw")

    assert result == {
        "error": "An error occurred while fetching https://example.com/page: direct timed out"
    }
