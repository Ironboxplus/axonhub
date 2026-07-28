package httpclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestHTTPClientAppliesConfiguredSSEEventLimitOverRealHTTP(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		_, _ = w.Write([]byte("data: " + strings.Repeat("x", 256) + "\n\n"))
	}))
	defer provider.Close()

	client := NewHttpClientWithClient(provider.Client(), WithMaxSSEEventSize(64))
	stream, err := client.DoStream(context.Background(), &Request{
		Method:  http.MethodPost,
		URL:     provider.URL,
		Headers: make(http.Header),
	})
	require.NoError(t, err)
	require.False(t, stream.Next())
	require.Error(t, stream.Err())
}
