package transformer

import (
	"net/url"
	"regexp"
	"strings"
)

var explicitAPIVersionSegment = regexp.MustCompile(`^v[0-9][A-Za-z0-9._-]*$`)

// NormalizeBaseURL normalizes the base URL for API endpoints.
// It ensures that the URL ends with the specified version and handles special cases:
//   - URLs ending with "#" have the marker stripped and skip version appending.
//     Endpoint paths such as "/responses" may still be appended later by BuildRequestURL.
//   - Trailing slashes are removed.
//   - Version is appended only if not already present.
//
// This is distinct from transformer-specific "##" handling, which enables true raw URL mode
// where no default endpoint path is appended.
func NormalizeBaseURL(rawURL, version string) string {
	if rawURL == "" {
		return ""
	}

	if before, ok := strings.CutSuffix(rawURL, "#"); ok {
		normalized := strings.TrimRight(before, "/")
		return normalized
	}

	if version == "" {
		return strings.TrimRight(rawURL, "/")
	}

	if strings.HasSuffix(rawURL, "/"+version) {
		return strings.TrimRight(rawURL, "/")
	}

	if strings.Contains(rawURL, "/"+version+"/") {
		return strings.TrimRight(rawURL, "/")
	}

	trimmed := strings.TrimRight(rawURL, "/")
	if version == "v1" && hasExplicitAPIVersion(trimmed) {
		return trimmed
	}

	return trimmed + "/" + version
}

func hasExplicitAPIVersion(rawURL string) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	for _, segment := range strings.Split(strings.Trim(parsed.Path, "/"), "/") {
		if explicitAPIVersionSegment.MatchString(segment) {
			return true
		}
	}
	return false
}

// BuildRequestURL constructs the full request URL from base URL and path parameters.
// When endpointPath is set (non-empty), it overrides defaultPath and skips default version
// normalization on the base URL itself (but maintains compatibility with "#" and "##" behaviors).
//
// Parameters:
//   - baseURL: the channel's base URL (may contain "#" or "##" suffix)
//   - version: the API version to append (e.g., "v1")
//   - defaultPath: the default endpoint path (e.g., "/chat/completions")
//   - endpointPath: optional custom endpoint path override from channel endpoint config
//   - rawURL: when true, use baseURL as-is without appending any path
func BuildRequestURL(baseURL, version, defaultPath, endpointPath string, rawURL bool) string {
	if rawURL {
		return strings.TrimRight(baseURL, "/")
	}

	normalized := NormalizeBaseURL(baseURL, version)

	if endpointPath != "" {
		return normalized + endpointPath
	}

	return normalized + defaultPath
}
