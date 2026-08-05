package emulation

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/emulation/hosted"
	"github.com/looplj/axonhub/llm/emulation/mcp"
)

type ledgerWebSearchService struct{}

func (ledgerWebSearchService) Search(context.Context, hosted.WebSearchInput) (hosted.WebSearchOutput, error) {
	return hosted.WebSearchOutput{Text: "ledger fixture"}, nil
}

func TestHostedExecutionLedgerEnforcesSourceMaxUses(t *testing.T) {
	t.Parallel()
	maxUses := int64(1)
	request := &llm.Request{ToolDefinitions: []llm.ToolDefinition{{
		Kind: llm.ToolKindWebSearch, LogicalName: "web_search", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{Type: "web_search", WebSearch: &llm.WebSearch{MaxUses: &maxUses}},
	}}}
	registry := ledgerHostedRegistry(t, request)
	binding, ok := registry.BindingFor(llm.ToolKindWebSearch, "web_search")
	require.True(t, ok)
	ledger := newHostedExecutionLedger(nil)
	require.NoError(t, ledger.reserve(context.Background(), []gatewayHostedCall{ledgerHostedCall(binding, "call_1", `{"query":"first"}`)}))
	err := ledger.reserve(context.Background(), []gatewayHostedCall{ledgerHostedCall(binding, "call_2", `{"query":"second"}`)})
	require.ErrorIs(t, err, ErrHostedMaxUses)
}

func TestHostedExecutionLedgerDoesNotCountPriorConversationHistory(t *testing.T) {
	t.Parallel()
	maxUses := int64(1)
	request := &llm.Request{ToolDefinitions: []llm.ToolDefinition{{
		Kind: llm.ToolKindWebSearch, LogicalName: "web_search", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{Type: "web_search", WebSearch: &llm.WebSearch{MaxUses: &maxUses}},
	}}, Input: []llm.Item{{
		Kind: llm.ItemKindHostedCall,
		HostedCall: &llm.HostedToolCall{
			Invocation: llm.ToolInvocation{Kind: llm.ToolKindWebFetch, LogicalName: "unrelated", CallID: "unrelated_1"},
			Result:     &llm.ToolResult{Kind: llm.ToolKindWebFetch, CallID: "unrelated_1", Status: llm.ToolResultStatusCompleted},
		},
	}, {
		Kind: llm.ItemKindHostedCall,
		HostedCall: &llm.HostedToolCall{
			Invocation: llm.ToolInvocation{
				Kind: llm.ToolKindWebSearch, LogicalName: "web_search", CallID: "history_1",
				ArgumentsJSON: json.RawMessage(`{"query":"history"}`),
			},
			Result: &llm.ToolResult{Kind: llm.ToolKindWebSearch, CallID: "history_1", Status: llm.ToolResultStatusCompleted},
		},
	}}}
	registry := ledgerHostedRegistry(t, request)
	binding, ok := registry.BindingFor(llm.ToolKindWebSearch, "web_search")
	require.True(t, ok)
	ledger := newHostedExecutionLedger(nil)
	require.NoError(t, ledger.reserve(context.Background(), []gatewayHostedCall{ledgerHostedCall(binding, "call_2", `{"query":"new"}`)}))
	require.False(t, ledger.allows(binding.Definition), "the current gateway transaction has now consumed max_uses")
}

func TestHostedExecutionLedgerCanonicalizesEquivalentJSONForRepeatDetection(t *testing.T) {
	t.Parallel()
	request := &llm.Request{ToolDefinitions: []llm.ToolDefinition{{
		Kind: llm.ToolKindWebSearch, LogicalName: "web_search", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{Type: "web_search", WebSearch: &llm.WebSearch{}},
	}}}
	registry := ledgerHostedRegistry(t, request)
	binding, ok := registry.BindingFor(llm.ToolKindWebSearch, "web_search")
	require.True(t, ok)
	ledger := newHostedExecutionLedger(map[llm.ToolKind]int{llm.ToolKindWebSearch: 4})
	require.NoError(t, ledger.reserve(context.Background(), []gatewayHostedCall{ledgerHostedCall(binding, "call_1", `{"query":"same","count":2}`)}))
	err := ledger.reserve(context.Background(), []gatewayHostedCall{ledgerHostedCall(binding, "call_2", `{"count":2,"query":"same"}`)})
	require.ErrorIs(t, err, ErrRepeatedHostedInvocation)
}

func TestHostedExecutionLedgerCriticalBranches(t *testing.T) {
	t.Parallel()
	require.NotNil(t, newHostedExecutionLedger(nil))

	maxUses := int64(5)
	definition := llm.ToolDefinition{
		Kind: llm.ToolKindWebSearch, LogicalName: "web_search",
		Hosted: &llm.HostedToolDefinition{WebSearch: &llm.WebSearch{MaxUses: &maxUses}},
	}
	ledger := &hostedExecutionLedger{operatorLimits: map[llm.ToolKind]int{llm.ToolKindWebSearch: 3}}
	require.Equal(t, 3, ledger.effectiveLimit(definition))
	ledger.operatorLimits[llm.ToolKindWebSearch] = 8
	require.Equal(t, 5, ledger.effectiveLimit(definition))

	require.Equal(t, []byte(`{"broken"`), canonicalHostedArguments(llm.ToolInvocation{ArgumentsJSON: json.RawMessage(`{"broken"`)}))
	require.Equal(t, []byte("plain text"), canonicalHostedArguments(llm.ToolInvocation{ArgumentsText: "  plain text  "}))
	require.Equal(t, []byte("input text"), canonicalHostedArguments(llm.ToolInvocation{InputText: "  input text  "}))
}

func TestGatewayFailureDiagnosticsAndProviderRoundContextCriticalBranches(t *testing.T) {
	t.Parallel()
	require.Equal(t, llm.ErrorDiagnostic{}, (*gatewayFailure)(nil).SafeDiagnostic())
	require.Equal(t, "hosted_max_uses", (&gatewayFailure{reason: GatewayStopHostedMaxUses}).SafeDiagnostic().Code)
	require.Equal(t, "hosted_repeated_invocation", (&gatewayFailure{reason: GatewayStopRepeatedInvocation}).SafeDiagnostic().Code)
	require.Equal(t, llm.ErrorDiagnostic{}, (&gatewayFailure{reason: "unknown"}).SafeDiagnostic())
	require.Equal(t, http.StatusGatewayTimeout, (&gatewayFailure{reason: GatewayStopProviderRoundTime}).SafeDiagnostic().StatusCode)

	controller := &Controller{}
	_, _, err := controller.providerRoundContext(nil, time.Time{}, true)
	require.ErrorIs(t, err, context.Canceled)
	background := context.Background()
	got, cancel, err := controller.providerRoundContext(background, time.Time{}, false)
	require.NoError(t, err)
	require.Equal(t, background, got)
	cancel()
	canceledCtx, canceledCancel := context.WithCancel(context.Background())
	canceledCancel()
	_, _, err = controller.providerRoundContext(canceledCtx, time.Time{}, true)
	require.ErrorIs(t, err, context.Canceled)
	controller.config.MaxProviderRoundTime = time.Nanosecond
	got, cancel, err = controller.providerRoundContext(background, time.Time{}, false)
	require.NoError(t, err)
	require.Equal(t, background, got, "the initial provider round is governed by the request deadline, not continuation limits")
	cancel()

	deadlineCtx, deadlineCancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer deadlineCancel()
	controller.config.DeadlineReserve = time.Second
	_, _, err = controller.providerRoundContext(deadlineCtx, time.Time{}, true)
	require.ErrorIs(t, err, ErrRequestDeadlineBudgetExhausted)
	require.NotErrorIs(t, err, ErrGatewayWallTimeExceeded)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Equal(t, "request_deadline_exceeded", llm.ErrorDiagnosticFrom(err).Code)
	controller.config.MaxRounds = 4
	stream := &controllerMCPStream{ctx: deadlineCtx, controller: controller, roundsDone: 1}
	err = stream.openRound()
	require.ErrorIs(t, err, ErrRequestDeadlineBudgetExhausted)
	require.Equal(t, "request_deadline_exceeded", llm.ErrorDiagnosticFrom(err).Code)

	gatewayDeadline := time.Now().Add(20 * time.Millisecond)
	gatewayCtx, gatewayCancel := context.WithDeadlineCause(context.Background(), gatewayDeadline, ErrGatewayWallTimeExceeded)
	defer gatewayCancel()
	_, _, err = controller.providerRoundContext(gatewayCtx, gatewayDeadline, true)
	require.ErrorIs(t, err, ErrGatewayWallTimeExceeded)
	require.NotErrorIs(t, err, ErrRequestDeadlineBudgetExhausted)
	controller.config.MaxProviderRoundTime = time.Second
	controller.config.DeadlineReserve = time.Millisecond
	longGatewayDeadline := time.Now().Add(20 * time.Millisecond)
	longGatewayCtx, longGatewayCancel := context.WithDeadlineCause(context.Background(), longGatewayDeadline, ErrGatewayWallTimeExceeded)
	defer longGatewayCancel()
	gatewayRoundCtx, gatewayRoundCancel, err := controller.providerRoundContext(longGatewayCtx, longGatewayDeadline, true)
	require.NoError(t, err)
	defer gatewayRoundCancel()
	select {
	case <-gatewayRoundCtx.Done():
	case <-time.After(200 * time.Millisecond):
		t.Fatal("gateway-owned provider budget did not expire")
	}
	require.ErrorIs(t, context.Cause(gatewayRoundCtx), ErrGatewayWallTimeExceeded)

	controller.config.MaxProviderRoundTime = time.Second
	controller.config.DeadlineReserve = 5 * time.Millisecond
	boundedCtx, boundedCancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer boundedCancel()
	roundCtx, roundCancel, err := controller.providerRoundContext(boundedCtx, time.Time{}, true)
	require.NoError(t, err)
	defer roundCancel()
	roundDeadline, ok := roundCtx.Deadline()
	require.True(t, ok)
	parentDeadline, ok := boundedCtx.Deadline()
	require.True(t, ok)
	require.Less(t, roundDeadline, parentDeadline)
	requestCauseCtx, requestCauseCancel := context.WithDeadlineCause(
		context.Background(), time.Now().Add(-time.Second), ErrRequestDeadlineBudgetExhausted,
	)
	defer requestCauseCancel()
	requestCauseErr := normalizeProviderRoundContextError(requestCauseCtx, context.DeadlineExceeded)
	require.ErrorIs(t, requestCauseErr, ErrRequestDeadlineBudgetExhausted)
	require.Equal(t, "request_deadline_exceeded", llm.ErrorDiagnosticFrom(requestCauseErr).Code)
	causeCtx, causeCancel := context.WithCancel(context.Background())
	causeCancel()
	require.ErrorIs(t, deadlineBudgetFailure(causeCtx, time.Now(), time.Time{}), context.Canceled)
	for _, test := range []struct {
		err  error
		want string
	}{
		{want: providerRoundOutcomeCompleted},
		{err: ErrGatewayProviderRoundTimeExceeded, want: providerRoundOutcomeProviderTimeout},
		{err: ErrGatewayWallTimeExceeded, want: providerRoundOutcomeGatewayWall},
		{err: ErrRequestDeadlineBudgetExhausted, want: providerRoundOutcomeRequestDeadline},
		{err: context.DeadlineExceeded, want: providerRoundOutcomeRequestDeadline},
		{err: context.Canceled, want: providerRoundOutcomeCanceled},
		{err: errors.New("transport"), want: providerRoundOutcomeTransportError},
	} {
		require.Equal(t, test.want, providerRoundOutcome(test.err))
	}
}

func TestProcessRoundDoesNotReserveHostedCallsFromInterruptedProviderLifecycle(t *testing.T) {
	t.Parallel()
	maxUses := int64(1)
	request := &llm.Request{ToolDefinitions: []llm.ToolDefinition{{
		Kind: llm.ToolKindWebSearch, LogicalName: "web_search", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{Type: "web_search", WebSearch: &llm.WebSearch{MaxUses: &maxUses}},
	}}}
	registry := ledgerHostedRegistry(t, request)
	binding, ok := registry.BindingFor(llm.ToolKindWebSearch, "web_search")
	require.True(t, ok)
	ledger := newHostedExecutionLedger(nil)
	interrupted := ledgerHostedCall(binding, "call_interrupted", `{"query":"first"}`)
	processed, err := (&Controller{}).processRound(
		context.Background(), mcp.NewEmptyRegistry(nil), registry, ledger, nil,
		&llm.Response{Status: llm.ResponseStatusIncomplete, Output: []llm.Item{interrupted.item}}, 1, false, false,
	)
	require.NoError(t, err)
	require.True(t, processed.boundary)
	require.True(t, ledger.allows(binding.Definition), "an interrupted provider lifecycle must not consume execution quota")
	require.NoError(t, ledger.reserve(context.Background(), []gatewayHostedCall{
		ledgerHostedCall(binding, "call_completed", `{"query":"first"}`),
	}))
}

func TestProcessRoundHostedBatchPreflightNeverPartiallyExecutes(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name      string
		maxUses   int64
		arguments []string
		wantErr   error
	}{
		{name: "same semantic invocation", maxUses: 4, arguments: []string{`{"query":"same","count":2}`, `{"count":2,"query":"same"}`}, wantErr: ErrRepeatedHostedInvocation},
		{name: "source max uses", maxUses: 1, arguments: []string{`{"query":"first"}`, `{"query":"second"}`}, wantErr: ErrHostedMaxUses},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			request := &llm.Request{ToolDefinitions: []llm.ToolDefinition{{
				Kind: llm.ToolKindWebSearch, LogicalName: "web_search", Execution: llm.ExecutionOwnerProvider,
				Hosted: &llm.HostedToolDefinition{Type: "web_search", WebSearch: &llm.WebSearch{MaxUses: &test.maxUses}},
			}}}
			registry := ledgerHostedRegistry(t, request)
			binding, ok := registry.BindingFor(llm.ToolKindWebSearch, "web_search")
			require.True(t, ok)
			ledger := newHostedExecutionLedger(nil)
			calls := []gatewayHostedCall{
				ledgerHostedCall(binding, "batch_1", test.arguments[0]),
				ledgerHostedCall(binding, "batch_2", test.arguments[1]),
			}
			_, err := (&Controller{}).processRound(
				context.Background(), mcp.NewEmptyRegistry(nil), registry, ledger, nil,
				&llm.Response{Status: llm.ResponseStatusCompleted, Output: []llm.Item{calls[0].item, calls[1].item}},
				2, false, true,
			)
			require.ErrorIs(t, err, test.wantErr)
			require.True(t, ledger.allows(binding.Definition), "failed batch preflight must not reserve or execute a prefix")
		})
	}
}

func ledgerHostedRegistry(t *testing.T, request *llm.Request) *hosted.Registry {
	t.Helper()
	executor, err := hosted.NewWebSearchExecutor(hosted.WebSearchExecutorConfig{Service: ledgerWebSearchService{}})
	require.NoError(t, err)
	registry, err := hosted.DiscoverRegistry(request, hosted.Config{
		SyntheticNameKey: []byte("ledger-hosted-key-32-bytes-long!!"),
		Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: executor},
	}, func(llm.ToolDefinition) bool { return true })
	require.NoError(t, err)
	return registry
}

func ledgerHostedCall(binding hosted.Binding, callID, arguments string) gatewayHostedCall {
	return gatewayHostedCall{
		binding: binding,
		item: llm.Item{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, LogicalName: binding.SyntheticName, CallID: callID,
			ArgumentsJSON: json.RawMessage(arguments), Execution: llm.ExecutionOwnerGateway,
		}},
	}
}
