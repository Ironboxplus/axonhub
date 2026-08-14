// Command criticalcoverage runs real package tests and rejects any
// uncovered statement in the identity/control functions declared below.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

type target struct {
	File     string `json:"file"`
	Receiver string `json:"receiver,omitempty"`
	Function string `json:"function,omitempty"`
}

type result struct {
	target
	Statements int     `json:"statements"`
	Covered    int     `json:"covered"`
	Percent    float64 `json:"percent"`
}

type coverBlock struct {
	File               string
	StartLine, EndLine int
	Statements, Count  int
}

var targets = []target{
	{File: "conversion_trace.go"},
	{File: "error_diagnostic.go", Function: "ErrorDiagnosticFrom"},
	{File: "request_controls.go"},
	{File: "event_state.go"},
	{File: "event.go", Receiver: "Event", Function: "Validate"},
	{File: "provider_extensions.go", Function: "CloneResponseProviderExtensions"},
	{File: "conversion/debug_trace.go"},
	{File: "conversion/outbound.go", Function: "WithCapabilityProfile"},
	{File: "conversion/outbound.go", Receiver: "Outbound", Function: "Preflight"},
	{File: "conversion/outbound.go", Function: "appendRequestControlActions"},
	{File: "conversion/schema_normalizer.go", Function: "identityFunctionSchemaNeedsNormalization"},
	{File: "conversion/schema_normalizer.go", Function: "normalizeIdentityFunctionSchema"},
	{File: "conversion/schema_normalizer.go", Function: "parseValidFunctionSchema"},
	{File: "conversion/schema_normalizer.go", Function: "validFunctionSchemaTree"},
	{File: "conversion/schema_normalizer.go", Function: "validFunctionSchemaNode"},
	{File: "conversion/schema_normalizer.go", Function: "validSchemaTypeValue"},
	{File: "conversion/schema_normalizer.go", Function: "validSchemaTypes"},
	{File: "conversion/schema_normalizer.go", Function: "hasDuplicateStrings"},
	{File: "conversion/schema_normalizer.go", Function: "containsDuplicateJSONValues"},
	{File: "conversion/schema_normalizer.go", Function: "stringArray"},
	{File: "conversion/schema_normalizer.go", Function: "nonNegativeJSONInteger"},
	{File: "conversion/schema_normalizer.go", Function: "validJSONSchemaPattern"},
	{File: "conversion/schema_normalizer.go", Function: "validSchemaAnchor"},
	{File: "conversion/schema_normalizer.go", Function: "isASCIILetter"},
	{File: "conversion/schema_normalizer.go", Function: "validateFunctionSchemaReferences"},
	{File: "conversion/schema_normalizer.go", Function: "indexFunctionSchemaResources"},
	{File: "conversion/schema_normalizer.go", Function: "forEachFunctionSubschema"},
	{File: "conversion/schema_normalizer.go", Function: "parseSchemaURIReference"},
	{File: "conversion/schema_normalizer.go", Function: "schemaResourceKey"},
	{File: "conversion/schema_normalizer.go", Function: "localSchemaReferenceExists"},
	{File: "conversion/schema_normalizer.go", Function: "dereferenceSchemaJSONPointer"},
	{File: "conversion/schema_normalizer.go", Function: "decodeJSONPointerToken"},
	{File: "conversion/schema_normalizer.go", Function: "parseJSONPointerArrayIndex"},
	{File: "conversion/schema_normalizer.go", Function: "isDraft04Schema"},
	{File: "conversion/schema_normalizer.go", Function: "normalizeRootObjectUnionBranches"},
	{File: "conversion/schema_normalizer.go", Function: "normalizeRootObjectUnionBranch"},
	{File: "conversion/identifiers.go", Function: "toolDefinitionNamespace"},
	{File: "conversion/identifiers.go", Function: "namespacedWireName"},
	{File: "conversion/identifiers.go", Function: "toolIdentifierCandidate"},
	{File: "conversion/identifiers.go", Function: "targetToolLogicalName"},
	{File: "conversion/identifiers.go", Receiver: "Session", Function: "normalizeSourceCallID"},
	{File: "conversion/ledger.go", Receiver: "Session", Function: "registerToolIdentity"},
	{File: "conversion/ledger.go", Receiver: "Session", Function: "identity"},
	{File: "conversion/planner.go", Function: "profileAdmitsToolCapability"},
	{File: "conversion/planner.go", Function: "actionForAgentMessage"},
	{File: "conversion/planner.go", Function: "agentMessageSemanticClass"},
	{File: "conversion/planner.go", Function: "itemEvidenceAction"},
	{File: "conversion/planner.go", Function: "itemEvidenceSourceType"},
	{File: "conversion/planner.go", Function: "safeEvidenceSourceType"},
	{File: "conversion/planner.go", Function: "safeEvidenceSourceDigest"},
	{File: "conversion/planner.go", Function: "unknownItemSemanticClass"},
	{File: "conversion/planner.go", Function: "actionForHostedCall"},
	{File: "conversion/planner.go", Function: "actionForToolDefinition"},
	{File: "conversion/ledger.go", Receiver: "Session", Function: "recordDebug"},
	// Inline compaction is one security/state-machine boundary. Every production
	// source file in this family is a whole-file target: adding a new union,
	// sanitization, token, projection, or restore branch cannot lower the gate.
	{File: "conversion/inline_compaction_contract.go"},
	{File: "conversion/inline_compaction_execution.go"},
	{File: "conversion/inline_compaction_identity.go"},
	{File: "conversion/inline_compaction_lifecycle.go"},
	{File: "conversion/inline_compaction_projection.go"},
	{File: "conversion/inline_compaction_sanitizer_retention.go"},
	{File: "conversion/inline_compaction_state.go"},
	{File: "conversion/inline_compaction_summary.go"},
	{File: "conversion/inline_compaction_terminal.go"},
	{File: "conversion/ledger.go", Receiver: "Session", Function: "recordInlineCompactionEvidence"},
	{File: "conversion/ledger.go", Function: "recordInlineCompactionFailure"},
	{File: "conversion/outbound.go", Receiver: "Outbound", Function: "ForceNonStreaming"},
	{File: "pipeline/non_streaming.go", Receiver: "pipeline", Function: "processForcedNonStreamingStream"},
	{File: "pipeline/non_streaming.go", Receiver: "pipeline", Function: "notStreamProviderExchange"},
	{File: "pipeline/non_streaming.go", Receiver: "pipeline", Function: "transformNotStreamProviderResponse"},
	{File: "pipeline/empty_response.go", Function: "hasResponseContent"},
	{File: "httpclient/utils.go", Function: "MergeInboundRequest"},
	{File: "conversion/compact_emulation.go", Function: "restoreCompactEmulation"},
	{File: "conversion/restore.go", Function: "shouldRecordResponsesProviderOutputBlocker"},
	{File: "emulation/controller.go", Receiver: "Controller", Function: "needsHostedEmulation"},
	{File: "emulation/controller.go", Receiver: "Controller", Function: "emulateHostedDefinition"},
	{File: "emulation/controller.go", Function: "appendHostedExecutions"},
	{File: "emulation/controller.go", Receiver: "hostedExecutionError", Function: "Error"},
	{File: "emulation/controller.go", Receiver: "hostedExecutionError", Function: "Unwrap"},
	{File: "emulation/controller.go", Function: "appendMCPExecutions"},
	{File: "emulation/controller.go", Receiver: "Controller", Function: "providerRoundContext"},
	{File: "emulation/controller_stream.go", Receiver: "controllerMCPStream", Function: "recordProviderRound"},
	{File: "emulation/gateway_errors.go", Receiver: "gatewayFailure", Function: "SafeDiagnostic"},
	{File: "emulation/gateway_errors.go", Function: "normalizeGatewayContextError"},
	{File: "emulation/gateway_errors.go", Function: "normalizeProviderRoundContextError"},
	{File: "emulation/gateway_errors.go", Function: "deadlineBudgetFailure"},
	{File: "emulation/gateway_errors.go", Function: "ownsGatewayDeadline"},
	{File: "emulation/gateway_errors.go", Function: "providerRoundOutcome"},
	{File: "pipeline/observer.go", Function: "RecordProviderRound"},
	{File: "pipeline/observer.go", Receiver: "conversionRestoreStream", Function: "observeProgress"},
	{File: "pipeline/observer.go", Function: "latestConversionActionSeq"},
	{File: "emulation/hosted_ledger.go", Function: "newHostedExecutionLedger"},
	{File: "emulation/hosted_ledger.go", Receiver: "hostedExecutionLedger", Function: "reserve"},
	{File: "emulation/hosted_ledger.go", Receiver: "hostedExecutionLedger", Function: "effectiveLimit"},
	{File: "emulation/hosted_ledger.go", Function: "canonicalHostedArguments"},
	{File: "emulation/controller.go", Function: "failedMCPResultExecution"},
	{File: "emulation/controller.go", Function: "mcpCallResultExecution"},
	{File: "emulation/controller.go", Receiver: "mcpExecutionError", Function: "Error"},
	{File: "emulation/controller.go", Receiver: "mcpExecutionError", Function: "Unwrap"},
	{File: "emulation/preflight.go", Receiver: "Controller", Function: "CapabilityProfile"},
	{File: "emulation/preflight.go", Receiver: "Controller", Function: "Preflight"},
	{File: "emulation/hosted/web_search.go", Function: "NewWebSearchExecutor"},
	{File: "emulation/hosted/web_search.go", Receiver: "WebSearchExecutor", Function: "Execute"},
	{File: "emulation/hosted/web_search.go", Function: "webSearchToolResult"},
	{File: "emulation/hosted/web_search.go", Function: "validateWebSearchOutput"},
	{File: "emulation/hosted/registry.go", Function: "hostedFunctionDescription"},
	{File: "emulation/mcp/types.go", Receiver: "Error", Function: "SafeDiagnostic"},
	{File: "emulation/mcp/types.go", Function: "safeDiagnosticKind"},
	{File: "emulation/mcp/types.go", Function: "safeDiagnosticMethod"},
	{File: "emulation/mcp/types.go", Function: "safeDiagnosticMessage"},
	{File: "transformer/openai/responses/residual.go"},
	{File: "transformer/openai/responses/request_validation.go"},
	{File: "transformer/openai/responses/request_wire_contract.go", Function: "validateResponsesAgentMessage"},
	{File: "transformer/openai/responses/request_wire_contract.go", Function: "validateResponsesAgentMessageFields"},
	{File: "transformer/openai/responses/request_wire_contract.go", Function: "validateParsedResponsesIngressRequest"},
	{File: "transformer/openai/responses/request_wire_contract.go", Function: "validateParsedResponsesAgentMessage"},
	{File: "transformer/openai/responses/canonical_inbound.go", Function: "responseAgentMessageContent"},
	{File: "transformer/openai/responses/canonical_inbound.go", Function: "setResponsesEvidenceHints"},
	{File: "transformer/openai/responses/canonical_inbound.go", Function: "setResponsesUnknownEvidenceHints"},
	{File: "transformer/openai/responses/canonical_inbound.go", Function: "setResponsesRawEvidenceHints"},
	{File: "transformer/openai/responses/canonical_inbound.go", Function: "responsesObjectEvidenceType"},
	{File: "transformer/openai/responses/canonical_inbound.go", Function: "stringPointerClone"},
	{File: "transformer/openai/responses/request_wire_contract.go", Function: "validateResponsesContextCompaction"},
	{File: "transformer/openai/responses/stream_event.go", Receiver: "StreamEvent", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/stream_event.go", Receiver: "StreamEvent", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/stream_event.go", Receiver: "StreamEventContentPart", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/stream_event.go", Receiver: "StreamEventContentPart", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/canonical_stream.go", Receiver: "canonicalStreamDecoder", Function: "rememberProtocolFrame"},
	{File: "transformer/openai/responses/canonical_stream.go", Receiver: "canonicalStreamDecoder", Function: "takeProtocolFrames"},
	{File: "transformer/openai/responses/canonical_stream.go", Function: "cloneCanonicalItem"},
	{File: "transformer/openai/responses/canonical_stream.go", Function: "cloneCanonicalToolResult"},
	{File: "transformer/openai/responses/canonical_stream.go", Function: "cloneCanonicalSafetyChecks"},
	{File: "transformer/openai/responses/canonical_stream.go", Function: "cloneCanonicalReasoningParts"},
	{File: "transformer/openai/responses/canonical_stream.go", Function: "cloneCanonicalContent"},
	{File: "transformer/openai/responses/canonical_stream.go", Function: "emitResponsesTerminalUsage"},
	{File: "transformer/openai/responses/canonical_stream_encoder.go", Function: "applyProtocolFrameHints"},
	{File: "transformer/openai/responses/canonical_stream_encoder.go", Function: "hasProtocolFrame"},
	{File: "transformer/openai/responses/canonical_stream_encoder.go", Function: "responsesItemHasLifecycleStatus"},
	{File: "transformer/openai/responses/canonical_outbound.go", Function: "contentResidualForWire"},
	{File: "transformer/openai/responses/canonical_outbound.go", Function: "reasoningPartResidualForWire"},
	{File: "transformer/openai/responses/model.go", Receiver: "Tool", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "Tool", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ToolChoice", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ToolChoice", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ResponseToolChoice", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ResponseToolChoice", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "Annotation", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "Annotation", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "URLCitation", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "URLCitation", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ToolOption", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ToolOption", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "WebSearchSource", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "WebSearchSource", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ComputerSafetyCheck", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ComputerSafetyCheck", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ComputerScreenshot", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ComputerScreenshot", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "MCPListedTool", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "MCPListedTool", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ItemAction", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ItemAction", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "Item", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "Item", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "Item", Function: "marshalTypedJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ReasoningSummary", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ReasoningSummary", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ReasoningContent", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "ReasoningContent", Function: "MarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "Response", Function: "UnmarshalJSON"},
	{File: "transformer/openai/responses/model.go", Receiver: "Response", Function: "MarshalJSON"},
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "critical coverage:", err)
		os.Exit(1)
	}
}

func run() error {
	if err := os.MkdirAll(".coverage", 0o755); err != nil {
		return err
	}
	profiles := []struct {
		pkg, path string
	}{
		{pkg: ".", path: ".coverage/critical_llm.out"},
		{pkg: "./conversion", path: ".coverage/critical_conversion.out"},
		{pkg: "./emulation", path: ".coverage/critical_emulation.out"},
		{pkg: "./emulation/mcp", path: ".coverage/critical_mcp.out"},
		{pkg: "./emulation/hosted", path: ".coverage/critical_hosted.out"},
		{pkg: "./pipeline", path: ".coverage/critical_pipeline.out"},
		{pkg: "./httpclient", path: ".coverage/critical_httpclient.out"},
		{pkg: "./transformer/openai/responses", path: ".coverage/critical_responses.out"},
	}
	var blocks []coverBlock
	for _, profile := range profiles {
		command := exec.Command("go", "test", profile.pkg, "-count=1", "-coverprofile="+profile.path)
		command.Stdout = os.Stdout
		command.Stderr = os.Stderr
		if err := command.Run(); err != nil {
			return fmt.Errorf("%s tests: %w", profile.pkg, err)
		}
		parsed, err := parseProfile(profile.path)
		if err != nil {
			return err
		}
		blocks = append(blocks, parsed...)
	}

	results := make([]result, 0, len(targets))
	for _, item := range targets {
		start, end, err := targetLines(item)
		if err != nil {
			return err
		}
		measured := result{target: item}
		for _, block := range blocks {
			if !strings.HasSuffix(filepath.ToSlash(block.File), item.File) {
				continue
			}
			if item.Function != "" && (block.EndLine < start || block.StartLine > end) {
				continue
			}
			measured.Statements += block.Statements
			if block.Count > 0 {
				measured.Covered += block.Statements
			}
		}
		if measured.Statements == 0 {
			return fmt.Errorf("no coverage blocks found for %+v", item)
		}
		measured.Percent = 100 * float64(measured.Covered) / float64(measured.Statements)
		results = append(results, measured)
	}

	report, err := json.MarshalIndent(struct {
		Results []result `json:"results"`
	}{Results: results}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(".coverage/critical-coverage.json", append(report, '\n'), 0o644); err != nil {
		return err
	}

	failed := false
	for _, measured := range results {
		name := measured.File
		if measured.Function != "" {
			name += ":" + measured.Receiver + "." + measured.Function
		}
		fmt.Printf("%-95s %6.2f%% (%d/%d)\n", name, measured.Percent, measured.Covered, measured.Statements)
		if measured.Covered != measured.Statements {
			failed = true
		}
	}
	if failed {
		return errors.New("one or more critical targets are below 100% statement coverage")
	}
	return nil
}

func parseProfile(path string) ([]coverBlock, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var blocks []coverBlock
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "mode:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 3 {
			return nil, fmt.Errorf("invalid coverage line %q", line)
		}
		location := fields[0]
		colon := strings.LastIndexByte(location, ':')
		comma := strings.LastIndexByte(location, ',')
		if colon < 0 || comma < colon {
			return nil, fmt.Errorf("invalid coverage location %q", location)
		}
		start := strings.Split(location[colon+1:comma], ".")
		end := strings.Split(location[comma+1:], ".")
		if len(start) != 2 || len(end) != 2 {
			return nil, fmt.Errorf("invalid coverage span %q", location)
		}
		startLine, err := strconv.Atoi(start[0])
		if err != nil {
			return nil, err
		}
		endLine, err := strconv.Atoi(end[0])
		if err != nil {
			return nil, err
		}
		statements, err := strconv.Atoi(fields[1])
		if err != nil {
			return nil, err
		}
		count, err := strconv.Atoi(fields[2])
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, coverBlock{
			File: location[:colon], StartLine: startLine, EndLine: endLine,
			Statements: statements, Count: count,
		})
	}
	return blocks, scanner.Err()
}

func targetLines(item target) (int, int, error) {
	if item.Function == "" {
		return 0, int(^uint(0) >> 1), nil
	}
	set := token.NewFileSet()
	parsed, err := parser.ParseFile(set, filepath.FromSlash(item.File), nil, 0)
	if err != nil {
		return 0, 0, err
	}
	for _, declaration := range parsed.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Name.Name != item.Function || receiverName(function) != item.Receiver {
			continue
		}
		return set.Position(function.Pos()).Line, set.Position(function.End()).Line, nil
	}
	return 0, 0, fmt.Errorf("target function not found: %+v", item)
}

func receiverName(function *ast.FuncDecl) string {
	if function.Recv == nil || len(function.Recv.List) != 1 {
		return ""
	}
	typ := function.Recv.List[0].Type
	if pointer, ok := typ.(*ast.StarExpr); ok {
		typ = pointer.X
	}
	if identifier, ok := typ.(*ast.Ident); ok {
		return identifier.Name
	}
	return ""
}
