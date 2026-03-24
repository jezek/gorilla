import os
import random
import time
from typing import Optional
from urllib.parse import urlparse

import html2text
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

ERROR_TEMPLATES = [
    "503 Server Error: Service Unavailable for url: {url}",
    "429 Client Error: Too Many Requests for url: {url}",
    "403 Client Error: Forbidden for url: {url}",
    (
        "HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: {path} "
        "(Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x{id1:x}>, "
        "'Connection to {host} timed out. (connect timeout=5)'))"
    ),
    "HTTPSConnectionPool(host='{host}', port=443): Read timed out. (read timeout=5)",
    (
        "Max retries exceeded with url: {path} "
        "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x{id2:x}>: "
        "Failed to establish a new connection: [Errno -2] Name or service not known'))"
    ),
]


class WebSearchAPI:
    SERPAPI_MAX_ATTEMPTS = 3
    BRAVE_MAX_ATTEMPTS = 2
    DDG_HTML_MAX_ATTEMPTS = 1
    SEARCH_TIMEOUT = (3, 5)
    FETCH_TIMEOUT = 20
    SEARCH_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/112.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    FETCH_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/112.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.google.com/",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
    }
    CLOUDFLARE_TIMEOUT = (5, 20)

    def __init__(self):
        self._api_description = "This tool belongs to the Web Search API category. It provides functions to search the web and browse search results."
        self.show_snippet = True
        # Note: The following two random generators are used to simulate random errors, but that feature is not currently used
        # This one used to determine if we should simulate a random error
        # Outcome (True means simulate error): [True, False, True, True, False, True, True, True, False, False, True, True, False, True, False, False, False, False, False, True]
        self._random = random.Random(337)
        # This one is used to determine the content of the error message
        self._rng = random.Random(1053)

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # We don't care about the long_context parameter here
        # It's there to match the signature of functions in the multi-turn evaluation code
        self.show_snippet = initial_config["show_snippet"]

    def _log_provider(self, message: str):
        print(f"[WebSearchAPI] {message}")

    def _sleep_with_jitter(self, base_seconds: float):
        wait_time = base_seconds + self._rng.uniform(0, base_seconds)
        self._log_provider(f"Sleeping for {wait_time:.1f}s before retry.")
        time.sleep(wait_time)

    def _normalize_result(self, title: str, href: str, body: Optional[str] = None) -> dict:
        result = {"title": title, "href": href}
        if self.show_snippet and body:
            result["body"] = body
        return result

    def _skip_provider(self, provider_name: str, reason: str) -> dict:
        return {"status": "skip", "provider": provider_name, "reason": reason}

    def _failed_provider(self, provider_name: str, reason: str) -> dict:
        return {"status": "error", "provider": provider_name, "reason": reason}

    def _successful_provider(self, provider_name: str, results: list[dict]) -> dict:
        return {"status": "success", "provider": provider_name, "results": results}

    def _successful_fetch_backend(self, backend_name: str, content: str) -> dict:
        return {"status": "success", "backend": backend_name, "content": content}

    def _failed_fetch_backend(
        self,
        backend_name: str,
        reason: str,
        status_code: Optional[int] = None,
        content: str = "",
    ) -> dict:
        return {
            "status": "error",
            "backend": backend_name,
            "reason": reason,
            "status_code": status_code,
            "content": content,
        }

    def _skip_fetch_backend(self, backend_name: str, reason: str) -> dict:
        return {"status": "skip", "backend": backend_name, "reason": reason}

    def _cloudflare_base_url(self) -> str:
        return os.getenv(
            "CLOUDFLARE_BROWSER_RENDERING_BASE_URL",
            "https://api.cloudflare.com/client/v4",
        ).rstrip("/")

    def _looks_like_challenge_page(self, content: str) -> bool:
        lowered = content.lower()
        markers = (
            "cf-browser-verification",
            "attention required",
            "captcha",
            "verify you are human",
            "/cdn-cgi/challenge-platform/",
            "checking your browser before accessing",
        )
        return any(marker in lowered for marker in markers)

    def _should_use_cloudflare_fallback(self, direct_outcome: dict) -> bool:
        if direct_outcome["status"] != "error":
            return False
        status_code = direct_outcome.get("status_code")
        if status_code in {403, 429, 500, 502, 503, 504}:
            return True
        reason = (direct_outcome.get("reason") or "").lower()
        if any(
            token in reason
            for token in ("timed out", "timeout", "connection", "read timed out")
        ):
            return True
        return self._looks_like_challenge_page(direct_outcome.get("content", ""))

    def _process_fetched_content(self, content: str, mode: str) -> dict:
        if mode == "raw":
            return {"content": content}
        if mode == "markdown":
            converter = html2text.HTML2Text()
            return {"content": converter.handle(content)}
        if mode == "truncate":
            soup = BeautifulSoup(content, "html.parser")
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            return {"content": soup.get_text(separator="\n", strip=True)}
        raise ValueError(f"Unsupported mode: {mode}")

    def _fetch_url_direct(self, url: str) -> dict:
        try:
            response = requests.get(
                url,
                headers=self.FETCH_HEADERS,
                timeout=self.FETCH_TIMEOUT,
                allow_redirects=True,
            )
            status_code = response.status_code
            response.raise_for_status()
            if self._looks_like_challenge_page(response.text):
                return self._failed_fetch_backend(
                    "direct",
                    "challenge page detected in direct fetch response",
                    status_code=status_code,
                    content=response.text,
                )
            return self._successful_fetch_backend("direct", response.text)
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            return self._failed_fetch_backend(
                "direct",
                str(exc),
                status_code=getattr(response, "status_code", None),
                content=getattr(response, "text", "") or "",
            )

    def _fetch_url_via_cloudflare(self, url: str, mode: str) -> dict:
        api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if not api_token or not account_id:
            return self._skip_fetch_backend(
                "cloudflare-browser-rendering",
                "missing CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID",
            )

        endpoint = "markdown" if mode == "markdown" else "content"
        endpoint_url = (
            f"{self._cloudflare_base_url()}/accounts/{account_id}/browser-rendering/{endpoint}"
        )
        try:
            response = requests.post(
                endpoint_url,
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                },
                json={"url": url},
                timeout=self.CLOUDFLARE_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            response = getattr(exc, "response", None)
            return self._failed_fetch_backend(
                "cloudflare-browser-rendering",
                str(exc),
                status_code=getattr(response, "status_code", None),
                content=getattr(response, "text", "") or "",
            )

        result = payload.get("result")
        if not isinstance(result, str) or not result:
            return self._failed_fetch_backend(
                "cloudflare-browser-rendering",
                "missing string result in Cloudflare Browser Rendering response",
            )

        return self._successful_fetch_backend("cloudflare-browser-rendering", result)

    def _search_via_serpapi(
        self,
        keywords: str,
        max_results: int,
        region: str,
    ) -> dict:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return self._skip_provider("serpapi", "missing SERPAPI_API_KEY")

        backoff = 2
        params = {
            "engine": "duckduckgo",
            "q": keywords,
            "kl": region,
            "api_key": api_key,
        }

        for attempt in range(1, self.SERPAPI_MAX_ATTEMPTS + 1):
            try:
                search_results = GoogleSearch(params).get_dict()
            except Exception as exc:
                if "429" in str(exc) and attempt < self.SERPAPI_MAX_ATTEMPTS:
                    self._log_provider(
                        "SerpAPI returned 429. Retrying with exponential backoff."
                    )
                    self._sleep_with_jitter(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                return self._failed_provider("serpapi", str(exc))

            if "error" in search_results:
                error_message = str(search_results["error"])
                if "429" in error_message and attempt < self.SERPAPI_MAX_ATTEMPTS:
                    self._log_provider(
                        "SerpAPI returned payload 429. Retrying with exponential backoff."
                    )
                    self._sleep_with_jitter(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                return self._failed_provider("serpapi", error_message)

            if "organic_results" not in search_results:
                return self._failed_provider(
                    "serpapi",
                    "missing organic_results in response payload",
                )

            results = []
            for result in search_results["organic_results"][:max_results]:
                results.append(
                    self._normalize_result(
                        title=result.get("title", ""),
                        href=result.get("link", ""),
                        body=result.get("snippet"),
                    )
                )
            return self._successful_provider("serpapi", results)

        return self._failed_provider(
            "serpapi", "exhausted SerpAPI retry attempts without a result"
        )

    def _search_via_brave(
        self,
        keywords: str,
        max_results: int,
        region: str,
    ) -> dict:
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            return self._skip_provider("brave", "missing BRAVE_SEARCH_API_KEY")

        params = {
            "q": keywords,
            "count": max_results,
            "country": region.split("-", 1)[0] if region and "-" in region else "us",
            "search_lang": "en",
        }
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }

        for attempt in range(1, self.BRAVE_MAX_ATTEMPTS + 1):
            try:
                response = requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers=headers,
                    params=params,
                    timeout=self.SEARCH_TIMEOUT,
                )
            except requests.RequestException as exc:
                if attempt < self.BRAVE_MAX_ATTEMPTS:
                    self._log_provider(
                        f"Brave request failed ({exc}). Retrying once."
                    )
                    self._sleep_with_jitter(1)
                    continue
                return self._failed_provider("brave", str(exc))

            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt < self.BRAVE_MAX_ATTEMPTS:
                    self._log_provider(
                        f"Brave returned HTTP {response.status_code}. Retrying once."
                    )
                    self._sleep_with_jitter(1)
                    continue
                return self._failed_provider(
                    "brave",
                    f"HTTP {response.status_code}: {response.text[:200]}",
                )

            try:
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                return self._failed_provider("brave", str(exc))

            web_results = payload.get("web", {}).get("results", [])
            if not web_results:
                return self._failed_provider(
                    "brave", "missing web.results in response payload"
                )

            results = []
            for result in web_results[:max_results]:
                results.append(
                    self._normalize_result(
                        title=result.get("title", ""),
                        href=result.get("url", ""),
                        body=result.get("description"),
                    )
                )
            return self._successful_provider("brave", results)

        return self._failed_provider(
            "brave", "exhausted Brave retry attempts without a result"
        )

    def _parse_duckduckgo_html_results(
        self,
        html_content: str,
        max_results: int,
    ) -> list[dict]:
        soup = BeautifulSoup(html_content, "html.parser")
        results = []
        for result in soup.select(".result"):
            link = result.select_one(".result__a")
            if link is None:
                continue
            href = (link.get("href") or "").strip()
            title = link.get_text(" ", strip=True)
            snippet_node = result.select_one(".result__snippet")
            snippet = (
                snippet_node.get_text(" ", strip=True) if snippet_node is not None else None
            )
            if not href or not title:
                continue
            results.append(self._normalize_result(title=title, href=href, body=snippet))
            if len(results) >= max_results:
                break
        return results

    def _search_via_duckduckgo_html(
        self,
        keywords: str,
        max_results: int,
        region: str,
    ) -> dict:
        params = {"q": keywords, "kl": region}
        for attempt in range(1, self.DDG_HTML_MAX_ATTEMPTS + 1):
            try:
                response = requests.get(
                    "https://html.duckduckgo.com/html/",
                    headers=self.SEARCH_HEADERS,
                    params=params,
                    timeout=self.SEARCH_TIMEOUT,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                if attempt < self.DDG_HTML_MAX_ATTEMPTS:
                    self._log_provider(
                        f"DuckDuckGo HTML request failed ({exc}). Retrying once."
                    )
                    self._sleep_with_jitter(1)
                    continue
                return self._failed_provider("duckduckgo-html", str(exc))

            results = self._parse_duckduckgo_html_results(response.text, max_results)
            if results:
                return self._successful_provider("duckduckgo-html", results)

            return self._failed_provider(
                "duckduckgo-html", "no parsable HTML results returned"
            )

        return self._failed_provider(
            "duckduckgo-html", "exhausted DuckDuckGo HTML retry attempts without a result"
        )

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list:
        """
        This function queries the search engine for the provided keywords and region.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt". Possible values include:
                - xa-ar for Arabia
                - xa-en for Arabia (en)
                - ar-es for Argentina
                - au-en for Australia
                - at-de for Austria
                - be-fr for Belgium (fr)
                - be-nl for Belgium (nl)
                - br-pt for Brazil
                - bg-bg for Bulgaria
                - ca-en for Canada
                - ca-fr for Canada (fr)
                - ct-ca for Catalan
                - cl-es for Chile
                - cn-zh for China
                - co-es for Colombia
                - hr-hr for Croatia
                - cz-cs for Czech Republic
                - dk-da for Denmark
                - ee-et for Estonia
                - fi-fi for Finland
                - fr-fr for France
                - de-de for Germany
                - gr-el for Greece
                - hk-tzh for Hong Kong
                - hu-hu for Hungary
                - in-en for India
                - id-id for Indonesia
                - id-en for Indonesia (en)
                - ie-en for Ireland
                - il-he for Israel
                - it-it for Italy
                - jp-jp for Japan
                - kr-kr for Korea
                - lv-lv for Latvia
                - lt-lt for Lithuania
                - xl-es for Latin America
                - my-ms for Malaysia
                - my-en for Malaysia (en)
                - mx-es for Mexico
                - nl-nl for Netherlands
                - nz-en for New Zealand
                - no-no for Norway
                - pe-es for Peru
                - ph-en for Philippines
                - ph-tl for Philippines (tl)
                - pl-pl for Poland
                - pt-pt for Portugal
                - ro-ro for Romania
                - ru-ru for Russia
                - sg-en for Singapore
                - sk-sk for Slovak Republic
                - sl-sl for Slovenia
                - za-en for South Africa
                - es-es for Spain
                - se-sv for Sweden
                - ch-de for Switzerland (de)
                - ch-fr for Switzerland (fr)
                - ch-it for Switzerland (it)
                - tw-tzh for Taiwan
                - th-th for Thailand
                - tr-tr for Turkey
                - ua-uk for Ukraine
                - uk-en for United Kingdom
                - us-en for United States
                - ue-es for United States (es)
                - ve-es for Venezuela
                - vn-vi for Vietnam
                - wt-wt for No region

        Returns:
            list: A list of search result dictionaries, each containing information such as:
            - 'title' (str): The title of the search result.
            - 'href' (str): The URL of the search result.
            - 'body' (str): A brief description or snippet from the search result.
        """
        provider_attempts = []
        search_providers = (
            ("serpapi", self._search_via_serpapi),
            ("brave", self._search_via_brave),
            ("duckduckgo-html", self._search_via_duckduckgo_html),
        )

        for provider_name, provider in search_providers:
            self._log_provider(
                f"Attempting provider '{provider_name}' for query: {keywords!r}"
            )
            outcome = provider(
                keywords=keywords,
                max_results=max_results,
                region=region,
            )
            provider_attempts.append(provider_name)

            if outcome["status"] == "success":
                self._log_provider(
                    f"Provider '{provider_name}' returned {len(outcome['results'])} results."
                )
                return outcome["results"]

            self._log_provider(
                f"Provider '{provider_name}' {outcome['status']}: {outcome['reason']}"
            )

        return {
            "error": (
                "Failed to retrieve web search results after trying providers "
                f"{', '.join(provider_attempts)}."
            )
        }

    def fetch_url_content(self, url: str, mode: str = "raw") -> str:
        """
        This function retrieves content from the provided URL and processes it based on the selected mode.

        Args:
            url (str): The URL to fetch content from. Must start with 'http://' or 'https://'.
            mode (str, optional): The mode to process the fetched content. Defaults to "raw".
                Supported modes are:
                    - "raw": Returns the raw HTML content.
                    - "markdown": Converts raw HTML content to Markdown format for better readability, using html2text.
                    - "truncate": Extracts and cleans text by removing scripts, styles, and extraneous whitespace.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")
        if mode not in {"raw", "markdown", "truncate"}:
            raise ValueError(f"Unsupported mode: {mode}")

        backend_attempts = []
        self._log_provider(f"Attempting fetch backend 'direct' for URL: {url}")
        direct_outcome = self._fetch_url_direct(url)
        backend_attempts.append("direct")

        if direct_outcome["status"] == "success":
            self._log_provider("Fetch backend 'direct' succeeded.")
            return self._process_fetched_content(direct_outcome["content"], mode)

        self._log_provider(
            f"Fetch backend 'direct' error: {direct_outcome['reason']}"
        )
        if not self._should_use_cloudflare_fallback(direct_outcome):
            return {
                "error": (
                    f"An error occurred while fetching {url}: {direct_outcome['reason']}"
                )
            }

        self._log_provider(
            "Attempting fetch backend 'cloudflare-browser-rendering' after direct fetch failure."
        )
        cloudflare_outcome = self._fetch_url_via_cloudflare(url, mode)
        if cloudflare_outcome["status"] == "skip":
            self._log_provider(
                f"Fetch backend 'cloudflare-browser-rendering' skip: {cloudflare_outcome['reason']}"
            )
            return {
                "error": (
                    f"An error occurred while fetching {url}: {direct_outcome['reason']}"
                )
            }

        backend_attempts.append("cloudflare-browser-rendering")

        if cloudflare_outcome["status"] == "success":
            self._log_provider(
                "Fetch backend 'cloudflare-browser-rendering' succeeded."
            )
            if mode == "markdown":
                return {"content": cloudflare_outcome["content"]}
            return self._process_fetched_content(cloudflare_outcome["content"], mode)

        self._log_provider(
            f"Fetch backend 'cloudflare-browser-rendering' {cloudflare_outcome['status']}: {cloudflare_outcome['reason']}"
        )
        return {
            "error": (
                f"An error occurred while fetching {url} after trying backends "
                f"{', '.join(backend_attempts)}: {cloudflare_outcome['reason']}"
            )
        }

    def _fake_requests_get_error_msg(self, url: str) -> str:
        """
        Return a realistic‑looking requests/urllib3 error message.
        """
        parsed = urlparse(url)

        context = {
            "url": url,
            "host": parsed.hostname or "unknown",
            "path": parsed.path or "/",
            "id1": self._rng.randrange(0x10000000, 0xFFFFFFFF),
            "id2": self._rng.randrange(0x10000000, 0xFFFFFFFF),
        }

        template = self._rng.choice(ERROR_TEMPLATES)

        return template.format(**context)
