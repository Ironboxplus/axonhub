package hosted

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"strings"
)

func enforceDomainFilters(target *url.URL, allowedDomains, blockedDomains []string) error {
	allowed, err := normalizeDomainRules(allowedDomains)
	if err != nil {
		return fmt.Errorf("invalid allowed domain filter: %w", err)
	}
	blocked, err := normalizeDomainRules(blockedDomains)
	if err != nil {
		return fmt.Errorf("invalid blocked domain filter: %w", err)
	}
	host := strings.TrimSuffix(strings.ToLower(target.Hostname()), ".")
	if host == "" {
		return errors.New("target URL has no hostname")
	}
	for _, rule := range blocked {
		if domainMatches(host, rule) {
			return fmt.Errorf("target domain %q is blocked", host)
		}
	}
	if len(allowed) == 0 {
		return nil
	}
	for _, rule := range allowed {
		if domainMatches(host, rule) {
			return nil
		}
	}
	return fmt.Errorf("target domain %q is not allowed", host)
}

func normalizeDomainRules(rules []string) ([]string, error) {
	normalized := make([]string, 0, len(rules))
	for _, raw := range rules {
		rule := strings.TrimSuffix(strings.ToLower(strings.TrimSpace(raw)), ".")
		if err := validateDomainRule(rule); err != nil {
			return nil, fmt.Errorf("%q: %w", raw, err)
		}
		normalized = append(normalized, rule)
	}
	return normalized, nil
}

func validateDomainRule(rule string) error {
	if rule == "" {
		return errors.New("domain is empty")
	}
	for _, character := range rule {
		if character > 127 {
			return errors.New("domain must use ASCII form")
		}
	}
	if net.ParseIP(rule) != nil {
		return nil
	}
	if len(rule) > 253 || strings.ContainsAny(rule, "/\\?#@*:") {
		return errors.New("domain must not contain a scheme, port, path, wildcard, or user information")
	}
	for _, label := range strings.Split(rule, ".") {
		if label == "" || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return errors.New("domain contains an invalid label")
		}
		for _, character := range label {
			if character != '-' && (character < 'a' || character > 'z') && (character < '0' || character > '9') {
				return errors.New("domain contains an invalid character")
			}
		}
	}
	return nil
}

func domainMatches(host, rule string) bool {
	if net.ParseIP(host) != nil || net.ParseIP(rule) != nil {
		return host == rule
	}
	return host == rule || strings.HasSuffix(host, "."+rule)
}
