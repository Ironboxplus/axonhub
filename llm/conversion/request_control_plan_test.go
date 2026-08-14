package conversion

import (
	"context"
	"errors"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

type dispatchProbeOutbound struct {
	called bool
}

func (outbound *dispatchProbeOutbound) APIFormat() llm.APIFormat {
	return llm.APIFormatOpenAIResponse
}

func (outbound *dispatchProbeOutbound) TransformRequest(context.Context, *llm.Request) (*httpclient.Request, error) {
	outbound.called = true
	return &httpclient.Request{}, nil
}

func (outbound *dispatchProbeOutbound) TransformResponse(context.Context, *httpclient.Response) (*llm.Response, error) {
	return nil, nil
}

func (outbound *dispatchProbeOutbound) TransformStream(context.Context, *httpclient.Request, streams.Stream[*httpclient.StreamEvent]) (streams.Stream[*llm.Response], error) {
	return nil, nil
}

func (outbound *dispatchProbeOutbound) TransformError(context.Context, *httpclient.Error) *llm.ResponseError {
	return nil
}

func (outbound *dispatchProbeOutbound) AggregateStreamChunks(context.Context, *httpclient.Request, []*httpclient.StreamEvent) ([]byte, llm.ResponseMeta, error) {
	return nil, llm.ResponseMeta{}, nil
}

func TestPlannerRejectsDuplicateRequestControlWithObjectEvidence(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Input: []llm.Item{
			{Kind: llm.ItemKindMessage, Role: llm.RoleUser},
			{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}},
			{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}},
		},
	}

	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	var controlErr *llm.RequestControlError
	if !errors.Is(err, ErrIncompletePlan) || !errors.As(err, &controlErr) {
		t.Fatalf("planner error = %T %v, want incomplete plan wrapping RequestControlError", err, err)
	}
	if plan == nil || plan.Complete() || plan.Summary.Unknown != 1 {
		t.Fatalf("control rejection plan = %#v", plan)
	}
	if plan.Debug == nil || plan.Debug.Mode != llm.ConversionEvidenceRequired || len(plan.Debug.Actions) != 1 {
		t.Fatalf("control rejection evidence = %#v", plan.Debug)
	}
	action := plan.Debug.Actions[0]
	if action.ObjectKind != string(ObjectRequestControl) || action.ObjectID != "input[2]" ||
		action.FieldPath != "input[2].type" || action.Stage != llm.ConversionStageRequestPlanning ||
		action.Strategy != string(StrategyRequestControlMultiplicity) || action.Reason != string(ReasonDuplicateRequestControl) ||
		action.Result != llm.ConversionResultUnknown || action.Severity != llm.ConversionSeverityCritical || action.Reversible {
		t.Fatalf("control rejection action = %#v", action)
	}
	if evidencePlan, ok := PlanFromError(err); !ok || evidencePlan != plan {
		t.Fatalf("PlanFromError = %#v, %v; want original plan", evidencePlan, ok)
	}
}

func TestPlannerNormalizesRequestControlBeforePlanningWithoutMutatingSource(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Input: []llm.Item{
			{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}},
			{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "hello"}}},
		},
	}

	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || plan == nil || !plan.Complete() || plan.Summary.Lowered != 1 {
		t.Fatalf("normalized control plan = %#v, err=%v", plan, err)
	}
	if len(plan.Actions) != 3 || plan.Actions[0].Strategy != StrategyInlineCompactionGateway || plan.Actions[1].Strategy != StrategyInlineCompactionGateway ||
		plan.Actions[2].Strategy != StrategyRequestControl || plan.Actions[2].Ref.ItemIndex != 0 ||
		plan.Actions[2].DestinationPath != "input[1]" || plan.Debug == nil || len(plan.Debug.Actions) != 3 ||
		plan.Debug.Actions[2].DestinationPath != "input[1]" {
		t.Fatalf("normalized control actions = %#v", plan.Actions)
	}
	if request.Input[0].Kind != llm.ItemKindCompactionTrigger || request.Input[1].Kind != llm.ItemKindMessage {
		t.Fatalf("planner mutated source order: %#v", request.Input)
	}
}

func TestOutboundRejectsDuplicateRequestControlBeforeProviderEncoder(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Input: []llm.Item{
			{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}},
			{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}},
		},
	}
	provider := &dispatchProbeOutbound{}
	outbound := NewOutbound(provider)

	preflightPlan, preflightErr := outbound.Preflight(request)
	if preflightErr == nil || preflightPlan == nil || preflightPlan.Debug == nil || provider.called {
		t.Fatalf("preflight = plan %#v, err=%v, provider_called=%v", preflightPlan, preflightErr, provider.called)
	}

	_, transformErr := outbound.TransformRequest(context.Background(), request)
	if transformErr == nil || provider.called {
		t.Fatalf("transform error=%v, provider_called=%v", transformErr, provider.called)
	}
	transformPlan, ok := PlanFromError(transformErr)
	if !ok || transformPlan.Debug == nil || len(transformPlan.Debug.Actions) != 1 ||
		transformPlan.Debug.Actions[0].FieldPath != "input[1].type" {
		t.Fatalf("transform rejection evidence = %#v, ok=%v", transformPlan, ok)
	}
}

func TestOutboundPreflightRejectsGatewayCompactionWithoutDurableCodec(t *testing.T) {
	t.Parallel()
	provider := &dispatchProbeOutbound{}
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Input: []llm.Item{
			{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "retain this"}}},
			{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}},
		},
	}
	plan, err := NewOutbound(provider).Preflight(request)
	var inlineErr *InlineCompactionError
	if plan == nil || !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionCodecMissing || provider.called {
		t.Fatalf("preflight plan=%#v err=%v inline=%#v provider_called=%v", plan, err, inlineErr, provider.called)
	}
}

func TestOutboundUsesOneCapabilityProfileForPreflightAndTransform(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindMCP, LogicalName: "inventory", Execution: llm.ExecutionOwnerProvider,
			MCP: &llm.MCPDefinition{
				ServerLabel: "inventory", ServerURL: "https://mcp.example.test/rpc",
				AllowedTools: &llm.MCPToolFilter{ToolNames: []string{"lookup"}},
			},
		}},
	}
	profile, _ := ProfileFor(llm.APIFormatOpenAIResponse)
	profile.ID += "+gateway"
	profile.NativeTools &^= CapabilityMCPTool
	profile.EmulatedTools |= CapabilityMCPTool
	provider := &dispatchProbeOutbound{}
	outbound := NewOutbound(provider, WithCapabilityProfile(profile))

	preflight, err := outbound.Preflight(request)
	if err != nil || preflight == nil || preflight.Summary.Emulated != 1 ||
		preflight.Target.ID != profile.ID || preflight.Actions[0].Strategy != StrategyMCPGateway {
		t.Fatalf("configured preflight = %#v, err=%v", preflight, err)
	}
	httpRequest, err := outbound.TransformRequest(context.Background(), request)
	if err != nil || !provider.called {
		t.Fatalf("configured transform error=%v, provider_called=%v", err, provider.called)
	}
	summary, ok := SummaryFromRequest(httpRequest)
	if !ok || summary.ProfileID != profile.ID || summary.Emulated != preflight.Summary.Emulated || summary.Native != preflight.Summary.Native {
		t.Fatalf("transform summary = %#v, ok=%v; preflight=%#v", summary, ok, preflight.Summary)
	}
}
