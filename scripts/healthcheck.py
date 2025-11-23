import sys
from urllib import request, error


URL = "http://localhost:8000/health"
TIMEOUT = 2


try:
    with request.urlopen(URL, timeout=TIMEOUT) as response:
        if response.status == 200:
            sys.exit(0)
        else:
            sys.exit(1)

except (error.URLError, OSError) as e:
    print(f"Healthcheck failed: {e}", file=sys.stderr)
    sys.exit(1)
