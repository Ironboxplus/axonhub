package conversion

import (
	"context"
	"strings"
	"time"

	"github.com/looplj/axonhub/llm"
)

type streamToolKey struct {
	choice int
	tool   int
}

type streamToolState struct {
	identity       toolIdentity
	hasIdentity    bool
	ref            ObjectRef
	callID         string
	providerCallID string
	providerName   string
	args           jsonValueAccumulator
	input          string
	finished       bool
	schemaPaths    []schemaOptionalPath
	schemaArgs     jsonValueAccumulator
	schemaFinished bool
}

// streamRestorer owns provider stream state for one request. It is consumed by
// a single stream iterator, so it needs no locks and never adds contention to
// shared transformers.
type streamRestorer struct {
	ctx            context.Context
	session        *Session
	tools          map[streamToolKey]*streamToolState
	order          []streamToolKey
	eventTools     map[string]*streamToolState
	sequenceOffset uint64
}

func newStreamRestorer(session *Session) *streamRestorer {
	return newStreamRestorerContext(context.Background(), session)
}

func newStreamRestorerContext(ctx context.Context, session *Session) *streamRestorer {
	if ctx == nil {
		ctx = context.Background()
	}
	return &streamRestorer{
		ctx:        ctx,
		session:    session,
		tools:      make(map[streamToolKey]*streamToolState),
		eventTools: make(map[string]*streamToolState),
	}
}

func (r *streamRestorer) restore(response *llm.Response) *llm.Response {
	if response == nil || r == nil || r.session == nil {
		return response
	}
	startedAt := time.Time{}
	if r.session.traceEnabled {
		startedAt = time.Now()
	}
	for choiceIndex := range response.Choices {
		choice := &response.Choices[choiceIndex]
		// Choice.Index is the stable wire identity. Slice position is not: some
		// providers emit sparse choices or reorder them between chunks.
		r.restoreMessage(choice.Index, choice.Message)
		r.restoreMessage(choice.Index, choice.Delta)
		if choice.FinishReason != nil {
			r.flushIncomplete(choice.Index, choice)
		}
	}
	r.restoreEvents(response)
	if r.session.traceEnabled {
		r.session.addRestoreNanos(time.Since(startedAt).Nanoseconds())
	}
	return response
}

func (r *streamRestorer) restoreEvents(response *llm.Response) {
	if response == nil || len(response.Events) == 0 {
		return
	}
	source := response.Events
	restored := source[:0]
	detached := false
	for index := range source {
		event := source[index]
		event.Sequence += r.sequenceOffset
		key, err := event.ItemRef.StableKey()
		if err != nil {
			restored = append(restored, event)
			continue
		}
		ref := eventToolRef(event.ItemRef)
		event.ItemRef.CallID = r.session.normalizeSourceCallID(event.ItemRef.CallID, llm.ConversionDirectionStream, ref)
		normalizeStreamSnapshotIdentifiers(event.Snapshot, r.session, ref)

		switch event.Kind {
		case llm.EventKindItemAdded:
			if event.Snapshot == nil || event.Snapshot.ToolCall == nil {
				restored = append(restored, event)
				continue
			}
			providerName := event.Snapshot.ToolCall.LogicalName
			providerNamespace := event.Snapshot.ToolCall.Namespace
			identity, hasIdentity := r.session.identity(providerName, providerNamespace)
			schemaPaths := r.session.schemaRestoration(providerName, providerNamespace)
			if !hasIdentity && len(schemaPaths) == 0 {
				restored = append(restored, event)
				continue
			}
			state := &streamToolState{
				identity: identity, hasIdentity: hasIdentity, callID: event.Snapshot.ToolCall.CallID, providerCallID: event.Snapshot.ToolCall.CallID,
				providerName: providerName,
				ref:          ref, schemaPaths: schemaPaths,
			}
			r.eventTools[key] = state
			if hasIdentity {
				restoreCanonicalToolSnapshot(event.Snapshot, state, "")
				recordIdentityRestore(r.session, llm.ConversionDirectionStream, state.ref, identity, true)
			}
			restored = append(restored, event)

		case llm.EventKindToolInputDelta:
			state := r.eventTools[key]
			if state == nil {
				restored = append(restored, event)
				continue
			}
			if len(state.schemaPaths) > 0 && !state.schemaFinished {
				fragment := event.Delta.ArgumentsJSON
				if fragment == "" || !state.schemaArgs.Add(fragment) {
					// Optional-field restoration needs the complete JSON value. Buffer
					// only affected strict calls; unrelated streams stay zero-copy.
					continue
				}
				rawArguments := []byte(state.schemaArgs.String())
				r.session.recordProviderArgumentBytes(state.providerCallID, state.providerName, rawArguments, "")
				restoredArguments, changed := stripOptionalNullArguments(rawArguments, state.schemaPaths)
				state.schemaFinished = true
				event.Delta.ArgumentsJSON = string(restoredArguments)
				if changed {
					r.session.recordDebug(llm.ConversionDirectionStream, state.ref, "restore", StrategySchemaNormalize, ReasonSemanticProjection, true)
				}
				restored = append(restored, event)
				continue
			}
			if !state.hasIdentity || state.identity.SourceKind == llm.ToolKindFunction {
				restored = append(restored, event)
				continue
			}
			if state.identity.SourceKind != llm.ToolKindCustom {
				restored = append(restored, event)
				continue
			}
			fragment := event.Delta.ArgumentsJSON
			if fragment == "" || state.finished || !state.args.Add(fragment) {
				// The lowering wrapper is an implementation detail. Do not leak its
				// partial JSON fragments to a custom-tool source protocol.
				continue
			}
			input, valid := restoreCustomInput(state.args.String(), state.callID, r.session, llm.ConversionDirectionStream, state.ref)
			if !valid {
				input = state.args.String()
				recordIdentityRestore(r.session, llm.ConversionDirectionStream, state.ref, state.identity, false)
			}
			state.finished = true
			state.input = input
			event.Delta.ArgumentsJSON = ""
			event.Delta.InputText = input
			restored = append(restored, event)

		case llm.EventKindToolInputDone:
			if state := r.eventTools[key]; state != nil {
				if len(state.schemaPaths) > 0 && !state.schemaFinished && state.schemaArgs.Len() > 0 {
					r.session.addRestoreMiss()
					r.session.recordDebug(llm.ConversionDirectionStream, state.ref, "restore_miss", StrategySchemaNormalize, ReasonNoStrategy, false)
				}
				if !state.finished && state.args.Len() > 0 {
					state.finished = true
					var valid bool
					state.input, valid = restoreCustomInput(state.args.String(), state.callID, r.session, llm.ConversionDirectionStream, state.ref)
					if valid && state.input != "" {
						// The target ended an incomplete JSON wrapper, so no earlier
						// free-form delta could be emitted safely. Insert one canonical
						// delta before done and shift subsequent canonical sequences.
						if !detached {
							copyOfRestored := make([]llm.Event, len(restored), len(source)+1)
							copy(copyOfRestored, restored)
							restored = copyOfRestored
							detached = true
						}
						delta := llm.Event{
							Kind: llm.EventKindToolInputDelta, Sequence: event.Sequence,
							ItemRef: event.ItemRef, Delta: llm.Delta{InputText: state.input},
						}
						restored = append(restored, delta)
						event.Sequence++
						r.sequenceOffset++
					}
					if !valid {
						recordIdentityRestore(r.session, llm.ConversionDirectionStream, state.ref, state.identity, false)
					}
				}
			}
			restored = append(restored, event)

		case llm.EventKindItemDone:
			state := r.eventTools[key]
			if state == nil || event.Snapshot == nil {
				restored = append(restored, event)
				continue
			}
			restoreInvocationSchemaArguments(event.Snapshot.ToolCall, r.session, llm.ConversionDirectionStream, state.ref)
			if event.Snapshot.ToolCall != nil && event.Snapshot.ToolCall.Status == llm.ToolCallStatusCompleted {
				publishName := state.providerName
				if publishName == "" {
					publishName = event.Snapshot.ToolCall.LogicalName
				}
				clientName := event.Snapshot.ToolCall.LogicalName
				r.session.publishProviderArgumentBytes(r.ctx, state.providerCallID, state.callID, publishName)
				if clientName != "" && clientName != publishName {
					r.session.publishProviderArgumentBytes(r.ctx, state.providerCallID, state.callID, clientName)
				}
			}
			if state.hasIdentity && !state.finished && state.args.Len() > 0 {
				state.input, state.finished = restoreCustomInput(state.args.String(), state.callID, r.session, llm.ConversionDirectionStream, state.ref)
				if !state.finished {
					state.input = state.args.String()
					state.finished = true
					recordIdentityRestore(r.session, llm.ConversionDirectionStream, state.ref, state.identity, false)
				}
			}
			if state.hasIdentity {
				restoreCanonicalToolSnapshot(event.Snapshot, state, state.input)
			}
			restored = append(restored, event)

		default:
			restored = append(restored, event)
		}
	}
	response.Events = restored
}

func restoreCanonicalToolSnapshot(item *llm.Item, state *streamToolState, input string) {
	if item == nil || item.ToolCall == nil || state == nil {
		return
	}
	if state.identity.SourceKind == llm.ToolKindFunction {
		item.ToolCall.LogicalName = state.identity.SourceName
		item.ToolCall.Namespace = state.identity.SourceNamespace
		return
	}
	if state.identity.SourceKind != llm.ToolKindCustom {
		item.ProtocolHints.SourceFormat = llm.APIFormatOpenAIResponse
		item.ProtocolHints.SourceType = sourceItemType(state.identity.SourceKind)
		item.ToolCall.Kind = state.identity.SourceKind
		item.ToolCall.LogicalName = state.identity.SourceName
		return
	}
	item.ProtocolHints.SourceFormat = llm.APIFormatOpenAIResponse
	item.ProtocolHints.SourceType = "custom_tool_call"
	item.ToolCall.Kind = llm.ToolKindCustom
	item.ToolCall.LogicalName = state.identity.SourceName
	item.ToolCall.ArgumentsJSON = nil
	item.ToolCall.ArgumentsText = ""
	item.ToolCall.InputText = input
}

func (r *streamRestorer) restoreMessage(choiceIndex int, message *llm.Message) {
	if message == nil {
		return
	}
	for callIndex := range message.ToolCalls {
		call := &message.ToolCalls[callIndex]
		providerCallID := call.ID
		key := streamToolKey{choice: choiceIndex, tool: call.Index}
		ref := messageToolRef(choiceIndex, call.Index)
		call.ID = r.session.normalizeSourceCallID(providerCallID, llm.ConversionDirectionStream, ref)
		if call.ResponseCustomToolCall != nil {
			call.ResponseCustomToolCall.CallID = r.session.normalizeSourceCallID(call.ResponseCustomToolCall.CallID, llm.ConversionDirectionStream, ref)
		}
		state := r.tools[key]
		providerName := call.Function.Name
		providerNamespace := call.Function.Namespace
		identity, hasIdentity := r.session.identity(providerName, providerNamespace)
		schemaPaths := r.session.schemaRestoration(providerName, providerNamespace)
		if hasIdentity || len(schemaPaths) > 0 {
			if state == nil {
				state = &streamToolState{
					identity: identity, hasIdentity: hasIdentity, ref: ref, schemaPaths: schemaPaths, providerCallID: providerCallID, providerName: providerName,
				}
				r.tools[key] = state
				r.order = append(r.order, key)
				if hasIdentity {
					recordIdentityRestore(r.session, llm.ConversionDirectionStream, state.ref, identity, true)
				}
			}
		}
		if state == nil {
			continue
		}
		if call.ID != "" {
			state.callID = call.ID
		}
		if len(state.schemaPaths) > 0 && call.Function.Arguments != "" && !state.schemaFinished {
			fragment := call.Function.Arguments
			call.Function.Arguments = ""
			if state.schemaArgs.Add(fragment) {
				rawArguments := []byte(state.schemaArgs.String())
				r.session.recordProviderArgumentBytes(state.providerCallID, call.Function.Name, rawArguments, "")
				restoredArguments, changed := stripOptionalNullArguments(rawArguments, state.schemaPaths)
				state.schemaFinished = true
				call.Function.Arguments = string(restoredArguments)
				if changed {
					r.session.recordDebug(llm.ConversionDirectionStream, state.ref, "restore", StrategySchemaNormalize, ReasonSemanticProjection, true)
				}
			}
		}
		if !state.hasIdentity {
			continue
		}
		if state.identity.SourceKind == llm.ToolKindFunction {
			call.Function.Name = state.identity.SourceName
			call.Function.Namespace = state.identity.SourceNamespace
			continue
		}
		if state.identity.SourceKind != llm.ToolKindCustom {
			call.Function.Name = state.identity.SourceName
			continue
		}

		input := ""
		if call.Function.Arguments != "" && !state.finished {
			if state.args.Add(call.Function.Arguments) {
				var valid bool
				input, valid = restoreCustomInput(state.args.String(), state.callID, r.session, llm.ConversionDirectionStream, state.ref)
				if !valid {
					input = state.args.String()
					recordIdentityRestore(r.session, llm.ConversionDirectionStream, state.ref, state.identity, false)
				}
				state.finished = true
			}
		}
		call.ID = state.callID
		call.Type = llm.ToolTypeResponsesCustomTool
		call.ResponseCustomToolCall = &llm.ResponseCustomToolCall{
			CallID: state.callID,
			Name:   state.identity.SourceName,
			Input:  input,
		}
		call.Function = llm.FunctionCall{}
	}
}

func normalizeStreamSnapshotIdentifiers(item *llm.Item, session *Session, ref ObjectRef) {
	if item == nil || session == nil {
		return
	}
	if item.ToolCall != nil {
		item.ToolCall.CallID = session.normalizeSourceCallID(item.ToolCall.CallID, llm.ConversionDirectionStream, ref)
	}
	if item.ToolResult != nil {
		item.ToolResult.CallID = session.normalizeSourceCallID(item.ToolResult.CallID, llm.ConversionDirectionStream, ref)
	}
	if item.HostedCall != nil {
		item.HostedCall.Invocation.CallID = session.normalizeSourceCallID(item.HostedCall.Invocation.CallID, llm.ConversionDirectionStream, ref)
		if item.HostedCall.Result != nil {
			item.HostedCall.Result.CallID = session.normalizeSourceCallID(item.HostedCall.Result.CallID, llm.ConversionDirectionStream, ref)
		}
	}
}

func (r *streamRestorer) flushIncomplete(choiceIndex int, choice *llm.Choice) {
	if choice == nil {
		return
	}
	// Creation order is the logical stream order. Iterating the map here made
	// terminal chunks nondeterministic when more than one call was incomplete.
	for _, key := range r.order {
		state := r.tools[key]
		if key.choice != choiceIndex || state == nil || state.finished || state.args.Len() == 0 {
			continue
		}
		if choice.Delta == nil {
			choice.Delta = &llm.Message{Role: "assistant"}
		}
		raw := state.args.String()
		input, valid := restoreCustomInput(raw, state.callID, r.session, llm.ConversionDirectionStream, state.ref)
		choice.Delta.ToolCalls = append(choice.Delta.ToolCalls, llm.ToolCall{
			Index: key.tool,
			ID:    state.callID,
			Type:  llm.ToolTypeResponsesCustomTool,
			ResponseCustomToolCall: &llm.ResponseCustomToolCall{
				CallID: state.callID,
				Name:   state.identity.SourceName,
				Input:  input,
			},
		})
		state.finished = true
		state.input = input
		if !valid {
			recordIdentityRestore(r.session, llm.ConversionDirectionStream, state.ref, state.identity, false)
		}
	}
}

func eventToolRef(ref llm.ItemRef) ObjectRef {
	result := ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
	if ref.OutputIndex != nil {
		result.ItemIndex = *ref.OutputIndex
	}
	if ref.ChoiceIndex != nil {
		result.MessageIndex = *ref.ChoiceIndex
	}
	if ref.ToolIndex != nil {
		result.ToolCallIndex = *ref.ToolIndex
	}
	return result
}

// jsonValueAccumulator recognizes the end of one streamed JSON value in one
// linear pass. Decoding happens once, only after the value is structurally
// complete, so large tool inputs do not incur repeated whole-buffer parsing.
type jsonValueAccumulator struct {
	raw       strings.Builder
	depth     int
	started   bool
	inString  bool
	escaped   bool
	complete  bool
	malformed bool
}

func (a *jsonValueAccumulator) Add(fragment string) bool {
	if a == nil || a.complete {
		return a != nil && a.complete
	}
	a.raw.WriteString(fragment)
	for i := 0; i < len(fragment); i++ {
		char := fragment[i]
		if !a.started {
			if char == ' ' || char == '\t' || char == '\r' || char == '\n' {
				continue
			}
			a.started = true
			if char != '{' && char != '[' {
				a.malformed = true
				continue
			}
			a.depth = 1
			continue
		}
		if a.malformed {
			continue
		}
		if a.inString {
			if a.escaped {
				a.escaped = false
				continue
			}
			switch char {
			case '\\':
				a.escaped = true
			case '"':
				a.inString = false
			}
			continue
		}
		switch char {
		case '"':
			a.inString = true
		case '{', '[':
			a.depth++
		case '}', ']':
			a.depth--
			if a.depth == 0 {
				a.complete = true
			}
			if a.depth < 0 {
				a.malformed = true
			}
		}
	}
	return a.complete
}

func (a *jsonValueAccumulator) String() string {
	if a == nil {
		return ""
	}
	return a.raw.String()
}

func (a *jsonValueAccumulator) Len() int {
	if a == nil {
		return 0
	}
	return a.raw.Len()
}
