import json
import re
import sys
import urllib.request
import urllib.error

SERPER_API_KEY = "YOUR_SERPER_API_KEY_HERE"
QUERY_FILE = "query.txt"
OUTPUT_FILE = "searchResults.txt"

# How many Serper results pages to fetch before giving up
MAX_PAGES = 5
RESULTS_PER_PAGE = 10


def is_statcan_pid_url(url: str) -> bool:
    """
    Returns True if the URL is from the StatCan domain AND ends with
    'pid=' followed only by digits (i.e. when read backwards, the first
    non-digit characters are '=', 'd', 'i', 'p').

    Examples that match:
        https://www23.statcan.gc.ca/imdb/p2SV.pl?Function=getSurvey&Id=1234&pid=98765
        https://www150.statcan.gc.ca/t1/tbl1/en/dtbl!96-325-X2021001/pid=9810028402
    """
    # Must be a StatCan domain
    if "statcan.gc.ca" not in url.lower():
        return False

    # Strip fragment, then check the URL up to (and not including) any '?'
    # We look at the full URL string reversed for the =dip pattern.
    url_no_fragment = url.split("#")[0]

    # Reverse the URL and look for the pattern =dip<digits>
    # i.e. the tail of the original URL is:  pid=<digits>
    reversed_url = url_no_fragment[::-1]
    pattern = re.compile(r"^\d*=dip", re.IGNORECASE)
    return bool(pattern.match(reversed_url))


def search_serper(query: str, page: int = 1) -> list[dict]:
    """Call the Serper /search endpoint and return organic results."""
    payload = json.dumps({"q": query, "num": RESULTS_PER_PAGE, "page": page}).encode()
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data.get("organic", [])
    except urllib.error.HTTPError as exc:
        print(f"HTTP error from Serper API: {exc.code} {exc.reason}", file=sys.stderr)
        return []
    except urllib.error.URLError as exc:
        print(f"Network error: {exc.reason}", file=sys.stderr)
        return []


def main():
    # Read query
    try:
        with open(QUERY_FILE, "r", encoding="utf-8") as f:
            query = f.read().strip()
    except FileNotFoundError:
        print(f"Error: '{QUERY_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    if not query:
        print("Error: query.txt is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Query: {query}")
    print(f"Searching for StatCan links ending in 'pid=<number>'...\n")

    matching_links: list[str] = []

    for page in range(1, MAX_PAGES + 1):
        print(f"Fetching page {page}...")
        results = search_serper(query, page=page)

        if not results:
            print("No more results returned.")
            break

        for result in results:
            link = result.get("link", "")
            if is_statcan_pid_url(link):
                print(f"  ✓ Match found: {link}")
                matching_links.append(link)

        # If we found at least one match we can stop (first result only)
        # Comment out the break below if you want ALL matches across pages
        if matching_links:
            break

    # Write results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        if matching_links:
            for link in matching_links:
                f.write(link + "\n")
            print(f"\n{len(matching_links)} link(s) written to '{OUTPUT_FILE}'.")
        else:
            f.write("No matching StatCan pid= links found.\n")
            print("\nNo matching links found. See searchResults.txt.")


if __name__ == "__main__":
    main()