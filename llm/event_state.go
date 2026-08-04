package llm

import "fmt"

type ResponseStreamState string

const (
	ResponseStreamStateInit       ResponseStreamState = "init"
	ResponseStreamStateStarted    ResponseStreamState = "started"
	ResponseStreamStateInProgress ResponseStreamState = "in_progress"
	ResponseStreamStateTerminal   ResponseStreamState = "terminal"
)

type StreamItemState string

const (
	StreamItemStateUnknown   StreamItemState = "unknown"
	StreamItemStateAdded     StreamItemState = "added"
	StreamItemStateStreaming StreamItemState = "streaming"
	StreamItemStateDone      StreamItemState = "done"
)

type StreamToolState string

const (
	StreamToolStateDeclared       StreamToolState = "declared"
	StreamToolStateInputStreaming StreamToolState = "input_streaming"
	StreamToolStateInputDone      StreamToolState = "input_done"
	StreamToolStateResultPending  StreamToolState = "result_pending"
	StreamToolStateHostedRunning  StreamToolState = "hosted_running"
	StreamToolStateResultDone     StreamToolState = "result_done"
)

type StreamInvariantCode string

const (
	StreamInvariantSequence      StreamInvariantCode = "sequence_violation"
	StreamInvariantResponseState StreamInvariantCode = "response_state_violation"
	StreamInvariantItemState     StreamInvariantCode = "item_state_violation"
	StreamInvariantToolState     StreamInvariantCode = "tool_state_violation"
	StreamInvariantOutputIndex   StreamInvariantCode = "output_index_collision"
	StreamInvariantOpenItems     StreamInvariantCode = "terminal_with_open_items"
	StreamInvariantInvalidEvent  StreamInvariantCode = "invalid_event"
)

// StreamInvariantError is safe to persist in conversion traces: it contains
// lifecycle metadata but never model text or tool arguments.
type StreamInvariantError struct {
	Code          StreamInvariantCode
	EventKind     EventKind
	Sequence      uint64
	ItemRef       ItemRef
	ResponseState ResponseStreamState
	ItemState     StreamItemState
	ToolState     StreamToolState
	Cause         error
}

func (err *StreamInvariantError) Error() string {
	if err == nil {
		return "<nil>"
	}
	message := fmt.Sprintf("stream invariant %s at sequence %d event %s", err.Code, err.Sequence, err.EventKind)
	if err.Cause != nil {
		message += ": " + err.Cause.Error()
	}
	return message
}

func (err *StreamInvariantError) Unwrap() error { return err.Cause }

type streamItemLifecycle struct {
	state StreamItemState
	tool  StreamToolState
	kind  ItemKind
}

// StreamStateMachine validates one request-owned canonical stream. It has no
// shared state and requires no locks. Protocol adapters must synthesize any
// lifecycle events their wire format omits before calling Apply.
type StreamStateMachine struct {
	response   ResponseStreamState
	terminal   EventKind
	items      map[string]streamItemLifecycle
	outputKeys map[int]string
	lastSeq    uint64
	hasLastSeq bool
}

func NewStreamStateMachine() *StreamStateMachine {
	return &StreamStateMachine{
		response:   ResponseStreamStateInit,
		items:      make(map[string]streamItemLifecycle),
		outputKeys: make(map[int]string),
	}
}

func (machine *StreamStateMachine) ResponseState() ResponseStreamState {
	if machine == nil {
		return ResponseStreamStateInit
	}
	return machine.response
}

func (machine *StreamStateMachine) TerminalKind() EventKind {
	if machine == nil {
		return ""
	}
	return machine.terminal
}

func (machine *StreamStateMachine) Apply(event Event) error {
	if machine == nil {
		return &StreamInvariantError{Code: StreamInvariantResponseState, EventKind: event.Kind, Sequence: event.Sequence, Cause: fmt.Errorf("nil state machine")}
	}
	if err := event.Validate(); err != nil {
		return machine.violation(StreamInvariantInvalidEvent, event, "", streamItemLifecycle{}, err)
	}
	if machine.hasLastSeq && event.Sequence <= machine.lastSeq {
		return machine.violation(StreamInvariantSequence, event, "", streamItemLifecycle{}, fmt.Errorf("sequence must increase after %d", machine.lastSeq))
	}
	machine.hasLastSeq = true
	machine.lastSeq = event.Sequence

	if machine.response == ResponseStreamStateTerminal {
		return machine.violation(StreamInvariantResponseState, event, "", streamItemLifecycle{}, fmt.Errorf("event after terminal %s", machine.terminal))
	}

	switch event.Kind {
	case EventKindResponseStarted:
		if machine.response != ResponseStreamStateInit {
			return machine.violation(StreamInvariantResponseState, event, "", streamItemLifecycle{}, fmt.Errorf("response already %s", machine.response))
		}
		machine.response = ResponseStreamStateStarted
	case EventKindResponseInProgress:
		if machine.response != ResponseStreamStateStarted && machine.response != ResponseStreamStateInProgress {
			return machine.violation(StreamInvariantResponseState, event, "", streamItemLifecycle{}, fmt.Errorf("response is %s", machine.response))
		}
		machine.response = ResponseStreamStateInProgress
	case EventKindItemAdded:
		return machine.addItem(event)
	case EventKindTextDelta, EventKindReasoningDelta, EventKindRefusalDelta:
		return machine.streamItem(event, false)
	case EventKindToolInputDelta:
		return machine.streamItem(event, true)
	case EventKindToolInputDone:
		return machine.finishToolInput(event)
	case EventKindHostedStatus:
		return machine.updateHostedTool(event)
	case EventKindMCPStatus:
		return machine.updateMCP(event)
	case EventKindItemDone:
		return machine.finishItem(event)
	case EventKindUsage:
		if !machine.responseActive() {
			return machine.violation(StreamInvariantResponseState, event, "", streamItemLifecycle{}, fmt.Errorf("response is %s", machine.response))
		}
	case EventKindError:
		// A transport or provider can reject before response.started. The error is
		// still a valid canonical event, but it does not pretend a response began.
	case EventKindResponseCompleted, EventKindResponseFailed,
		EventKindResponseIncomplete, EventKindResponseCancelled:
		return machine.finishResponse(event)
	}
	return nil
}

func (machine *StreamStateMachine) addItem(event Event) error {
	if !machine.responseActive() {
		return machine.violation(StreamInvariantResponseState, event, "", streamItemLifecycle{}, fmt.Errorf("response is %s", machine.response))
	}
	key, _ := event.ItemRef.key()
	if current, exists := machine.items[key]; exists && current.state != StreamItemStateUnknown {
		return machine.violation(StreamInvariantItemState, event, key, current, fmt.Errorf("item already %s", current.state))
	}
	if event.ItemRef.OutputIndex != nil {
		outputIndex := *event.ItemRef.OutputIndex
		if existingKey, exists := machine.outputKeys[outputIndex]; exists && existingKey != key {
			return machine.violation(
				StreamInvariantOutputIndex,
				event,
				key,
				streamItemLifecycle{},
				fmt.Errorf("output index %d already belongs to %s", outputIndex, existingKey),
			)
		}
		machine.outputKeys[outputIndex] = key
	}
	lifecycle := streamItemLifecycle{state: StreamItemStateAdded, kind: event.Snapshot.Kind}
	if event.Snapshot.Kind == ItemKindToolCall || event.Snapshot.Kind == ItemKindMCPCall {
		lifecycle.tool = StreamToolStateDeclared
	} else if event.Snapshot.Kind == ItemKindHostedCall || event.Snapshot.Kind == ItemKindMCPListTools {
		lifecycle.tool = StreamToolStateDeclared
	}
	machine.items[key] = lifecycle
	return nil
}

func (machine *StreamStateMachine) streamItem(event Event, toolInput bool) error {
	key, _ := event.ItemRef.key()
	current, exists := machine.items[key]
	if !exists || (current.state != StreamItemStateAdded && current.state != StreamItemStateStreaming) {
		return machine.violation(StreamInvariantItemState, event, key, current, fmt.Errorf("delta requires an added item"))
	}
	if toolInput {
		if current.kind != ItemKindToolCall && current.kind != ItemKindMCPCall {
			return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("tool input delta targets %s", current.kind))
		}
		if current.tool != StreamToolStateDeclared && current.tool != StreamToolStateInputStreaming {
			return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("tool is %s", current.tool))
		}
		current.tool = StreamToolStateInputStreaming
	}
	current.state = StreamItemStateStreaming
	machine.items[key] = current
	return nil
}

func (machine *StreamStateMachine) finishToolInput(event Event) error {
	key, _ := event.ItemRef.key()
	current, exists := machine.items[key]
	if !exists || current.kind != ItemKindToolCall && current.kind != ItemKindMCPCall {
		return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("tool input done requires a declared tool item"))
	}
	if current.tool != StreamToolStateDeclared && current.tool != StreamToolStateInputStreaming {
		return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("tool is %s", current.tool))
	}
	current.tool = StreamToolStateInputDone
	current.state = StreamItemStateStreaming
	machine.items[key] = current
	return nil
}

func (machine *StreamStateMachine) updateHostedTool(event Event) error {
	key, _ := event.ItemRef.key()
	current, exists := machine.items[key]
	if !exists || current.kind != ItemKindHostedCall {
		return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("hosted status requires a hosted tool item"))
	}
	switch event.Delta.HostedStatus {
	case "queued", "in_progress", "searching", "generating":
		if current.tool != StreamToolStateDeclared && current.tool != StreamToolStateHostedRunning {
			return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("hosted tool is %s", current.tool))
		}
		current.tool = StreamToolStateHostedRunning
	case "completed", "failed":
		if current.tool != StreamToolStateDeclared && current.tool != StreamToolStateHostedRunning {
			return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("hosted tool is %s", current.tool))
		}
		current.tool = StreamToolStateResultDone
	default:
		return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("unknown hosted status %q", event.Delta.HostedStatus))
	}
	current.state = StreamItemStateStreaming
	machine.items[key] = current
	return nil
}

func (machine *StreamStateMachine) updateMCP(event Event) error {
	key, _ := event.ItemRef.key()
	current, exists := machine.items[key]
	if !exists || current.kind != ItemKindMCPCall && current.kind != ItemKindMCPListTools {
		return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("MCP status requires an MCP call or list item"))
	}

	status := MCPCallStatus(event.Delta.MCPStatus)
	switch current.kind {
	case ItemKindMCPCall:
		switch status {
		case MCPCallStatusCalling, MCPCallStatusInProgress:
			if current.tool != StreamToolStateInputDone && current.tool != StreamToolStateHostedRunning {
				return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("MCP call input is %s", current.tool))
			}
			current.tool = StreamToolStateHostedRunning
		case MCPCallStatusCompleted, MCPCallStatusIncomplete, MCPCallStatusFailed:
			if current.tool != StreamToolStateInputDone && current.tool != StreamToolStateHostedRunning {
				return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("MCP call is %s", current.tool))
			}
			current.tool = StreamToolStateResultDone
		default:
			return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("unknown MCP call status %q", status))
		}
	case ItemKindMCPListTools:
		switch status {
		case MCPCallStatusInProgress:
			if current.tool != StreamToolStateDeclared && current.tool != StreamToolStateHostedRunning {
				return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("MCP list is %s", current.tool))
			}
			current.tool = StreamToolStateHostedRunning
		case MCPCallStatusCompleted, MCPCallStatusIncomplete, MCPCallStatusFailed:
			if current.tool != StreamToolStateDeclared && current.tool != StreamToolStateHostedRunning {
				return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("MCP list is %s", current.tool))
			}
			current.tool = StreamToolStateResultDone
		default:
			return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("unknown MCP list status %q", status))
		}
	}
	current.state = StreamItemStateStreaming
	machine.items[key] = current
	return nil
}

func (machine *StreamStateMachine) finishItem(event Event) error {
	key, _ := event.ItemRef.key()
	current, exists := machine.items[key]
	if !exists || (current.state != StreamItemStateAdded && current.state != StreamItemStateStreaming) {
		return machine.violation(StreamInvariantItemState, event, key, current, fmt.Errorf("item done requires an added item"))
	}
	if current.kind != event.Snapshot.Kind {
		return machine.violation(StreamInvariantItemState, event, key, current, fmt.Errorf("snapshot kind changed from %s to %s", current.kind, event.Snapshot.Kind))
	}
	if current.kind == ItemKindToolCall && current.tool != StreamToolStateInputDone {
		return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("tool item done while input is %s", current.tool))
	}
	if current.kind == ItemKindHostedCall && current.tool != StreamToolStateResultDone {
		return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("hosted item done while result is %s", current.tool))
	}
	if (current.kind == ItemKindMCPCall || current.kind == ItemKindMCPListTools) && current.tool != StreamToolStateResultDone {
		return machine.violation(StreamInvariantToolState, event, key, current, fmt.Errorf("MCP item done while lifecycle is %s", current.tool))
	}
	current.state = StreamItemStateDone
	machine.items[key] = current
	return nil
}

func (machine *StreamStateMachine) finishResponse(event Event) error {
	if !machine.responseActive() {
		return machine.violation(StreamInvariantResponseState, event, "", streamItemLifecycle{}, fmt.Errorf("response is %s", machine.response))
	}
	for key, current := range machine.items {
		if current.state != StreamItemStateDone {
			return machine.violation(StreamInvariantOpenItems, event, key, current, fmt.Errorf("item is %s", current.state))
		}
	}
	machine.response = ResponseStreamStateTerminal
	machine.terminal = event.Kind
	return nil
}

func (machine *StreamStateMachine) responseActive() bool {
	return machine.response == ResponseStreamStateStarted || machine.response == ResponseStreamStateInProgress
}

func (machine *StreamStateMachine) violation(code StreamInvariantCode, event Event, _ string, lifecycle streamItemLifecycle, cause error) error {
	return &StreamInvariantError{
		Code:          code,
		EventKind:     event.Kind,
		Sequence:      event.Sequence,
		ItemRef:       event.ItemRef,
		ResponseState: machine.response,
		ItemState:     lifecycle.state,
		ToolState:     lifecycle.tool,
		Cause:         cause,
	}
}
