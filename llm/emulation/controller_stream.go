package emulation

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/emulation/hosted"
	"github.com/looplj/axonhub/llm/emulation/mcp"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/streams"
)

type controllerRoundItem struct {
	sourceKey   string
	publicRef   llm.ItemRef
	outputIndex int
	gateway     bool
	constraint  bool
	held        bool
	buffered    []llm.Event
}

// controllerMCPStream is a pull-based stream rewriter. It consumes one real
// provider stream at a time, suppresses provider-private response terminals
// and synthetic MCP functions, and presents one continuous canonical response
// to the source protocol encoder. No worker goroutine or unbounded channel is
// needed: downstream backpressure directly controls provider reads.
type controllerMCPStream struct {
	ctx         context.Context
	cancel      context.CancelFunc
	controller  *Controller
	registry    *mcp.Registry
	hosted      *hosted.Registry
	constraints *customConstraintRegistry
	rounds      pipeline.CanonicalRoundTripper
	prepared    *llm.Request

	provider    streams.Stream[*llm.Response]
	accumulator *llm.CanonicalResponseAccumulator
	roundItems  map[string]*controllerRoundItem
	roundOrder  []*controllerRoundItem
	gatewaySeen bool

	initial           []llm.Item
	usage             *llm.Usage
	totalCalls        int
	resumedCalls      int
	roundsDone        int
	constraintRetries int

	machine        *llm.StreamStateMachine
	sequence       uint64
	nextOutput     int
	publicStarted  bool
	terminalError  *llm.ResponseError
	terminalReason string

	id      string
	model   string
	object  string
	created int64

	queue   []*llm.Response
	current *llm.Response
	err     error
	done    bool
	closed  bool
	once    sync.Once
}

func (controller *Controller) streamMCP(ctx context.Context, request *llm.Request, rounds pipeline.CanonicalRoundTripper) (streams.Stream[*llm.Response], error) {
	var loopCtx context.Context
	var cancel context.CancelFunc
	if controller.config.MaxWallTime > 0 {
		loopCtx, cancel = context.WithTimeout(ctx, controller.config.MaxWallTime)
	} else {
		// A streamed tool loop owns work that may outlive a provider round. It
		// always needs a cancellable child so client Close can interrupt an MCP
		// or hosted executor even when no wall-time limit is configured.
		loopCtx, cancel = context.WithCancel(ctx)
	}
	gatewayRequest, err := projectGatewayAllowedTools(request, rounds.TargetFormat())
	if err != nil {
		cancel()
		pipeline.RecordEmulationFailure(loopCtx)
		return nil, err
	}
	constraints, err := compileCustomConstraints(loopCtx, gatewayRequest, rounds.TargetFormat())
	if err != nil {
		cancel()
		pipeline.RecordEmulationFailure(loopCtx)
		return nil, err
	}

	registry := mcp.NewEmptyRegistry(controller.config.MCP.SyntheticNameKey)
	if needsMCPEmulation(gatewayRequest, rounds.TargetFormat()) {
		registry, err = mcp.DiscoverRegistry(loopCtx, gatewayRequest, controller.config.MCP)
		if err != nil {
			_ = constraints.Close(context.WithoutCancel(loopCtx))
			cancel()
			pipeline.RecordEmulationFailure(loopCtx)
			return nil, err
		}
	}
	closeOnError := func() {
		_ = registry.Close(context.WithoutCancel(loopCtx))
		_ = constraints.Close(context.WithoutCancel(loopCtx))
		cancel()
	}
	prepared := gatewayRequest.Clone()
	resumed := []llm.Item(nil)
	if needsMCPEmulation(gatewayRequest, rounds.TargetFormat()) {
		prepared, resumed, err = controller.lowerHistory(loopCtx, gatewayRequest, registry)
		if err != nil {
			closeOnError()
			pipeline.RecordEmulationFailure(loopCtx)
			return nil, err
		}
	}
	hostedRegistry, err := hosted.DiscoverRegistry(gatewayRequest, controller.config.Hosted, func(definition llm.ToolDefinition) bool {
		return emulateHostedDefinition(gatewayRequest.APIFormat, rounds.TargetFormat(), definition)
	})
	if err != nil {
		closeOnError()
		pipeline.RecordEmulationFailure(loopCtx)
		return nil, err
	}
	prepared, err = lowerHostedHistory(prepared, hostedRegistry)
	if err != nil {
		closeOnError()
		pipeline.RecordEmulationFailure(loopCtx)
		return nil, err
	}
	if len(resumed) > controller.config.MaxToolCalls {
		closeOnError()
		pipeline.RecordEmulationLimit(loopCtx)
		return nil, ErrLoopLimit
	}

	stream := &controllerMCPStream{
		ctx: loopCtx, cancel: cancel, controller: controller, registry: registry, hosted: hostedRegistry,
		constraints: constraints, rounds: rounds, prepared: prepared,
		initial:    append(uniqueCurrentListItems(request.Input, registry.ListItems()), resumed...),
		totalCalls: len(resumed), resumedCalls: len(resumed),
		machine: llm.NewStreamStateMachine(),
	}
	stream.resetRound()
	if err := stream.openRound(); err != nil {
		stream.closeResources()
		pipeline.RecordEmulationFailure(loopCtx)
		return nil, err
	}
	return stream, nil
}

func (stream *controllerMCPStream) Next() bool {
	if stream == nil || stream.closed {
		return false
	}
	for len(stream.queue) == 0 && !stream.done && stream.err == nil {
		stream.drive()
	}
	if len(stream.queue) == 0 {
		stream.current = nil
		return false
	}
	stream.current = stream.queue[0]
	stream.queue[0] = nil
	stream.queue = stream.queue[1:]
	return true
}

func (stream *controllerMCPStream) Current() *llm.Response { return stream.current }

func (stream *controllerMCPStream) Err() error {
	if stream == nil {
		return nil
	}
	return stream.err
}

func (stream *controllerMCPStream) Close() error {
	if stream == nil {
		return nil
	}
	stream.closed = true
	return stream.closeResources()
}

func (stream *controllerMCPStream) drive() {
	if stream.provider == nil {
		if err := stream.openRound(); err != nil {
			stream.fail(err)
		}
		return
	}
	if !stream.provider.Next() {
		err := stream.provider.Err()
		_ = stream.provider.Close()
		stream.provider = nil
		if err != nil {
			stream.fail(err)
			return
		}
		if !stream.accumulator.IsTerminal() {
			stream.fail(errors.New("canonical provider stream ended without a terminal event"))
		}
		return
	}

	chunk := stream.provider.Current()
	if chunk == nil {
		return
	}
	// Canonical decoders may attach the terminal lifecycle event to the
	// provider's [DONE] object when no separate usage frame exists. Only a
	// marker with no canonical payload is ignorable.
	if (chunk == llm.DoneResponse || chunk.Object == "[DONE]") && len(chunk.Events) == 0 {
		return
	}
	if chunk.ID != "" {
		stream.id = chunk.ID
	}
	if chunk.Model != "" {
		stream.model = chunk.Model
	}
	if chunk.Object != "" {
		stream.object = chunk.Object
	}
	if chunk.Created != 0 {
		stream.created = chunk.Created
	}
	if err := stream.accumulator.Observe(chunk); err != nil {
		stream.fail(err)
		return
	}

	terminal := false
	for index := range chunk.Events {
		event := chunk.Events[index]
		switch event.Kind {
		case llm.EventKindResponseCompleted, llm.EventKindResponseIncomplete,
			llm.EventKindResponseCancelled, llm.EventKindResponseFailed:
			terminal = true
			stream.terminalReason = event.TerminalReason
			if event.Error != nil {
				copyError := *event.Error
				stream.terminalError = &copyError
			}
			continue
		}
		if err := stream.rewriteProviderEvent(event); err != nil {
			stream.fail(err)
			return
		}
	}
	if terminal {
		_ = stream.provider.Close()
		stream.provider = nil
		if err := stream.finishRound(); err != nil {
			stream.fail(err)
		}
	}
}

func (stream *controllerMCPStream) openRound() error {
	if stream.done || stream.err != nil {
		return nil
	}
	if stream.roundsDone >= stream.controller.config.MaxRounds {
		pipeline.RecordEmulationLimit(stream.ctx)
		return ErrLoopLimit
	}
	provider, err := stream.rounds.Stream(stream.ctx, stream.prepared)
	if err != nil {
		return err
	}
	stream.provider = provider
	return nil
}

func (stream *controllerMCPStream) resetRound() {
	stream.accumulator = llm.NewCanonicalResponseAccumulator()
	stream.roundItems = make(map[string]*controllerRoundItem)
	stream.roundOrder = nil
	stream.gatewaySeen = false
	stream.terminalError = nil
	stream.terminalReason = ""
}

func (stream *controllerMCPStream) rewriteProviderEvent(event llm.Event) error {
	switch event.Kind {
	case llm.EventKindResponseStarted:
		return stream.startPublic()
	case llm.EventKindResponseInProgress, llm.EventKindUsage:
		return nil
	case llm.EventKindError:
		if !stream.publicStarted {
			if err := stream.startPublic(); err != nil {
				return err
			}
		}
		return stream.emit(event)
	case llm.EventKindItemAdded:
		if !stream.publicStarted {
			if err := stream.startPublic(); err != nil {
				return err
			}
		}
		return stream.addRoundItem(event)
	case llm.EventKindItemDone, llm.EventKindTextDelta, llm.EventKindReasoningDelta,
		llm.EventKindRefusalDelta, llm.EventKindToolInputDelta, llm.EventKindToolInputDone,
		llm.EventKindHostedStatus, llm.EventKindMCPStatus:
		key, err := event.ItemRef.StableKey()
		if err != nil {
			return err
		}
		item := stream.roundItems[key]
		if item == nil {
			return fmt.Errorf("provider event references an unknown round item")
		}
		if item.gateway {
			return nil
		}
		remapped := cloneControllerEvent(event)
		remapped.ItemRef = item.publicRef
		if item.held {
			item.buffered = append(item.buffered, remapped)
			return nil
		}
		return stream.emit(remapped)
	default:
		return fmt.Errorf("unsupported provider event %q in gateway stream", event.Kind)
	}
}

func (stream *controllerMCPStream) addRoundItem(event llm.Event) error {
	if event.Snapshot == nil {
		return errors.New("provider item_added has no snapshot")
	}
	key, err := event.ItemRef.StableKey()
	if err != nil {
		return err
	}
	if _, duplicate := stream.roundItems[key]; duplicate {
		return errors.New("provider stream added the same round item twice")
	}
	ref := llm.ItemRef{ItemID: event.ItemRef.ItemID, CallID: event.ItemRef.CallID}
	if ref.ItemID == "" {
		ref.ItemID = event.Snapshot.ID
	}
	if ref.CallID == "" && event.Snapshot.ToolCall != nil {
		ref.CallID = event.Snapshot.ToolCall.CallID
	}
	gateway := false
	constraint := false
	if event.Snapshot.Kind == llm.ItemKindToolCall && event.Snapshot.ToolCall != nil {
		_, gateway = stream.registry.Binding(event.Snapshot.ToolCall.LogicalName)
		if !gateway {
			_, gateway = stream.hosted.Binding(event.Snapshot.ToolCall.LogicalName)
		}
		constraint = stream.constraints.Has(event.Snapshot.ToolCall.LogicalName) && event.Snapshot.ToolCall.Kind == llm.ToolKindCustom
	}
	held := stream.gatewaySeen || gateway || constraint
	outputIndex := -1
	if !held {
		outputIndex = stream.nextOutput
		stream.nextOutput++
		ref.OutputIndex = intPointer(outputIndex)
	}
	item := &controllerRoundItem{
		sourceKey: key, publicRef: ref, outputIndex: outputIndex,
		gateway: gateway, constraint: constraint, held: held && !gateway,
	}
	stream.roundItems[key] = item
	stream.roundOrder = append(stream.roundOrder, item)
	if gateway || constraint {
		stream.gatewaySeen = true
	}
	if gateway {
		return nil
	}
	remapped := cloneControllerEvent(event)
	remapped.ItemRef = ref
	if item.held {
		item.buffered = append(item.buffered, remapped)
		return nil
	}
	return stream.emit(remapped)
}

func (stream *controllerMCPStream) startPublic() error {
	if stream.publicStarted {
		return nil
	}
	stream.publicStarted = true
	if err := stream.emit(llm.Event{Kind: llm.EventKindResponseStarted}); err != nil {
		return err
	}
	if err := stream.emit(llm.Event{Kind: llm.EventKindResponseInProgress}); err != nil {
		return err
	}
	for index := range stream.initial {
		outputIndex := stream.nextOutput
		stream.nextOutput++
		if err := stream.emitMaterializedItem(stream.initial[index], outputIndex); err != nil {
			return err
		}
	}
	stream.initial = nil
	return nil
}

func (stream *controllerMCPStream) finishRound() error {
	if !stream.accumulator.IsTerminal() {
		return errors.New("provider round has no canonical terminal")
	}
	response := stream.accumulator.Snapshot()
	if response == nil {
		return errors.New("provider round has no canonical response")
	}
	stream.roundsDone++
	stream.usage = addUsage(stream.usage, response.Usage)
	processed, err := stream.controller.processRound(
		stream.ctx, stream.registry, stream.hosted, stream.constraints, response,
		stream.controller.config.MaxToolCalls-stream.totalCalls,
		stream.constraintRetries < stream.controller.config.MaxCustomConstraintRetries,
	)
	if err != nil {
		if errors.Is(err, ErrLoopLimit) {
			pipeline.RecordEmulationLimit(stream.ctx)
		}
		return err
	}
	stream.totalCalls += processed.gatewayCalls
	if processed.constraintRetry {
		stream.constraintRetries++
		pipeline.RecordCustomConstraintRetry(stream.ctx)
	}
	stream.usage = addUsage(stream.usage, usageForServerTools(processed.serverToolUsage))
	pipeline.RecordEmulationRound(
		stream.ctx, uint32(processed.gatewayCalls+stream.resumedCalls), uint32(processed.approvals),
	)
	stream.resumedCalls = 0

	if len(response.Output) != len(stream.roundOrder) {
		return fmt.Errorf("provider materialized %d items from %d stream lifecycles", len(response.Output), len(stream.roundOrder))
	}
	for index, state := range stream.roundOrder {
		item := response.Output[index]
		if state.gateway {
			if item.ToolCall == nil {
				return errors.New("gateway round item lost its tool call payload")
			}
			replacement, ok := processed.replacements[item.ToolCall.CallID]
			if !ok {
				return errors.New("gateway round item has no public replacement")
			}
			outputIndex := stream.assignHeldOutput(state)
			if err := stream.emitMaterializedItem(replacement, outputIndex); err != nil {
				return err
			}
			continue
		}
		if state.constraint && item.ToolCall != nil {
			if replacement, ok := processed.replacements[item.ToolCall.CallID]; ok {
				if replacement.Kind != "" {
					outputIndex := stream.assignHeldOutput(state)
					if err := stream.emitMaterializedItem(replacement, outputIndex); err != nil {
						return err
					}
				}
				continue
			}
		}
		if state.held {
			stream.assignHeldOutput(state)
			for eventIndex := range state.buffered {
				state.buffered[eventIndex].ItemRef = state.publicRef
				if err := stream.emit(state.buffered[eventIndex]); err != nil {
					return err
				}
			}
		}
	}

	if stream.controller.shouldContinuePauseTurn(stream.prepared, stream.rounds.TargetFormat(), response, processed) {
		appendProviderRound(stream.prepared, response.Output, processed.results)
		refreshRegistryDefinitions(stream.prepared, stream.registry)
		refreshHostedDefinitions(stream.prepared, stream.hosted)
		if stream.roundsDone >= stream.controller.config.MaxRounds {
			pipeline.RecordEmulationLimit(stream.ctx)
			return ErrLoopLimit
		}
		stream.resetRound()
		return nil
	}
	if processed.gatewayCalls == 0 || processed.boundary || response.Status != llm.ResponseStatusCompleted {
		return stream.finishPublic(response.Status, response.TerminalReason)
	}
	appendProviderRound(stream.prepared, response.Output, processed.results)
	refreshRegistryDefinitions(stream.prepared, stream.registry)
	refreshHostedDefinitions(stream.prepared, stream.hosted)
	if stream.roundsDone >= stream.controller.config.MaxRounds {
		pipeline.RecordEmulationLimit(stream.ctx)
		return ErrLoopLimit
	}
	stream.resetRound()
	return nil
}

func (stream *controllerMCPStream) assignHeldOutput(item *controllerRoundItem) int {
	if item.outputIndex >= 0 {
		return item.outputIndex
	}
	item.outputIndex = stream.nextOutput
	stream.nextOutput++
	item.publicRef.OutputIndex = intPointer(item.outputIndex)
	return item.outputIndex
}

func (stream *controllerMCPStream) emitMaterializedItem(item llm.Item, outputIndex int) error {
	response := &llm.Response{Output: []llm.Item{llm.CloneCanonicalItem(item)}, Status: llm.ResponseStatusCompleted}
	events, err := llm.CanonicalEventsFromResponse(response)
	if err != nil {
		return err
	}
	for index := range events {
		event := events[index]
		switch event.Kind {
		case llm.EventKindResponseStarted, llm.EventKindResponseInProgress,
			llm.EventKindUsage, llm.EventKindResponseCompleted:
			continue
		}
		event.ItemRef.OutputIndex = intPointer(outputIndex)
		if err := stream.emit(event); err != nil {
			return err
		}
	}
	return nil
}

func (stream *controllerMCPStream) finishPublic(status llm.ResponseStatus, terminalReason string) error {
	if !stream.publicStarted {
		if err := stream.startPublic(); err != nil {
			return err
		}
	}
	if stream.usage != nil {
		if err := stream.emit(llm.Event{Kind: llm.EventKindUsage, Usage: cloneUsage(stream.usage)}); err != nil {
			return err
		}
	}
	terminal := llm.Event{Kind: llm.EventKindResponseCompleted}
	terminal.TerminalReason = terminalReason
	switch status {
	case llm.ResponseStatusIncomplete:
		terminal.Kind = llm.EventKindResponseIncomplete
	case llm.ResponseStatusCancelled:
		terminal.Kind = llm.EventKindResponseCancelled
	case llm.ResponseStatusFailed:
		terminal.Kind = llm.EventKindResponseFailed
		terminal.Error = stream.terminalError
		if terminal.Error == nil {
			terminal.Error = publicEmulationError()
		}
	}
	if err := stream.emit(terminal); err != nil {
		return err
	}
	stream.done = true
	return stream.closeResources()
}

func (stream *controllerMCPStream) fail(err error) {
	if err == nil || stream.done || stream.err != nil {
		return
	}
	pipeline.RecordEmulationFailure(stream.ctx)
	if !stream.publicStarted {
		stream.err = err
		stream.done = true
		_ = stream.closeResources()
		return
	}
	publicError := publicEmulationError()
	if emitErr := stream.emit(llm.Event{Kind: llm.EventKindError, Error: publicError}); emitErr != nil {
		stream.err = errors.Join(err, emitErr)
		stream.done = true
		_ = stream.closeResources()
		return
	}
	if emitErr := stream.emit(llm.Event{Kind: llm.EventKindResponseFailed, Error: publicError}); emitErr != nil {
		stream.err = errors.Join(err, emitErr)
	}
	stream.done = true
	_ = stream.closeResources()
}

func (stream *controllerMCPStream) emit(event llm.Event) error {
	event.Sequence = stream.sequence
	stream.sequence++
	if err := stream.machine.Apply(event); err != nil {
		return err
	}
	stream.queue = append(stream.queue, &llm.Response{
		ID: stream.id, Model: stream.model, Object: stream.object, Created: stream.created,
		// This stream is reconstructed by the gateway controller and carries no
		// source-protocol wire payload. Leaving APIFormat unset is intentional:
		// source encoders must consume the canonical events instead of mistaking
		// them for an identity-protocol compatibility chunk.
		Events: []llm.Event{event},
	})
	return nil
}

func (stream *controllerMCPStream) closeResources() error {
	var result error
	stream.once.Do(func() {
		if stream.provider != nil {
			result = errors.Join(result, stream.provider.Close())
			stream.provider = nil
		}
		result = errors.Join(result, stream.registry.Close(context.WithoutCancel(stream.ctx)))
		result = errors.Join(result, stream.constraints.Close(context.WithoutCancel(stream.ctx)))
		stream.cancel()
	})
	return result
}

func cloneControllerEvent(event llm.Event) llm.Event {
	clone := event
	if event.Snapshot != nil {
		item := llm.CloneCanonicalItem(*event.Snapshot)
		clone.Snapshot = &item
	}
	clone.Delta.ProviderData = append([]byte(nil), event.Delta.ProviderData...)
	if event.Usage != nil {
		clone.Usage = cloneUsage(event.Usage)
	}
	if event.Error != nil {
		errorCopy := *event.Error
		clone.Error = &errorCopy
	}
	return clone
}

func publicEmulationError() *llm.ResponseError {
	return &llm.ResponseError{Detail: llm.ErrorDetail{
		Code: "gateway_emulation_failed", Type: "gateway_error", Message: "gateway emulation failed",
	}}
}

func intPointer(value int) *int { return &value }
