package hosted

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
)

func policyHTTPClient(
	label string,
	rawEndpoint string,
	baseClient *http.Client,
	policy EndpointPolicy,
) (*url.URL, *http.Client, error) {
	endpoint, err := url.Parse(rawEndpoint)
	if err != nil || !isAbsoluteHTTPURL(endpoint) {
		return nil, nil, fmt.Errorf("%s requires an absolute HTTP endpoint", label)
	}
	if policy == nil {
		return nil, nil, fmt.Errorf("%s requires an endpoint policy", label)
	}
	if err := policy(endpoint); err != nil {
		return nil, nil, fmt.Errorf("%s endpoint rejected: %w", label, err)
	}
	if baseClient == nil {
		baseClient = http.DefaultClient
	}
	client := *baseClient
	originalRedirect := baseClient.CheckRedirect
	client.CheckRedirect = func(request *http.Request, via []*http.Request) error {
		if !isAbsoluteHTTPURL(request.URL) {
			return fmt.Errorf("%s redirect rejected: invalid HTTP endpoint", label)
		}
		if err := policy(request.URL); err != nil {
			return fmt.Errorf("%s redirect rejected: %w", label, err)
		}
		if originalRedirect != nil {
			return originalRedirect(request, via)
		}
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}
		return nil
	}
	return endpoint, &client, nil
}

func isAbsoluteHTTPURL(candidate *url.URL) bool {
	return candidate != nil && (candidate.Scheme == "http" || candidate.Scheme == "https") &&
		candidate.Host != "" && candidate.User == nil && candidate.Fragment == ""
}
