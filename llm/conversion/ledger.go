package conversion

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/looplj/axonhub/llm"
)

type toolIdentity struct {
	SourceKind      llm.ToolKind
	SourceName      string
	SourceNamespace string
}

type schemaRestoration struct {
	paths          []schemaOptionalPath
	originalHash   [sha256.Size]byte
	normalizedHash [sha256.Size]byte
}

type providerArgumentRecordKey struct {
	callID string
	name   string
}

type compactEmulationState struct {
	instructions string
}

type Session struct {
	plan *Plan

	bySourceName          map[string]string
	bySyntheticName       map[string]toolIdentity
	occupiedNames         map[string]struct{}
	targetToolNames       *identifierLedger
	targetCallIDs         *identifierLedger
	sourceCallIDs         *identifierLedger
	schemaRestorations    map[string]schemaRestoration
	providerArgumentBytes map[providerArgumentRecordKey]json.RawMessage
	continuation          ContinuationBinding
	lowerNanos            atomic.Int64
	restoreNanos          atomic.Int64
	restoreMiss           atomic.Uint32
	customInputsRepaired  atomic.Uint32
	customInputRepairSeen sync.Map
	customInputRawSeen    sync.Map
	identifiersNormalized atomic.Uint32
	schemasNormalized     atomic.Uint32
	streamViolations      atomic.Uint32
	lastStreamViolation   atomic.Value
	lastStreamEvent       atomic.Value
	terminalEvent         atomic.Value
	traceEnabled          bool
	compactEmulation      *compactEmulationState
}

func newSession(plan *Plan, request *llm.Request, traceEnabled bool) *Session {
	session := &Session{
		plan:                  plan,
		bySourceName:          make(map[string]string),
		bySyntheticName:       make(map[string]toolIdentity),
		occupiedNames:         make(map[string]struct{}),
		schemaRestorations:    make(map[string]schemaRestoration),
		providerArgumentBytes: make(map[providerArgumentRecordKey]json.RawMessage),
		traceEnabled:          traceEnabled,
	}
	if request != nil {
		for index := range request.ToolDefinitions {
			definition := &request.ToolDefinitions[index]
			if definition.Kind != llm.ToolKindFunction || definition.Function == nil || definition.Function.Namespace == "" {
				continue
			}
			sourceName := strings.TrimPrefix(definition.LogicalName, definition.Function.Namespace+"__")
			session.bySyntheticName[definition.LogicalName] = toolIdentity{
				SourceKind: llm.ToolKindFunction, SourceName: sourceName, SourceNamespace: definition.Function.Namespace,
			}
		}
		for _, tool := range request.Tools {
			if tool.Type == llm.ToolTypeFunction && tool.Function.Name != "" {
				session.occupiedNames[tool.Function.Name] = struct{}{}
			}
		}
	}
	return session
}

func (s *Session) registerSchemaRestoration(targetName string, paths []schemaOptionalPath, original, normalized []byte) {
	if s == nil || targetName == "" {
		return
	}
	copied := make([]schemaOptionalPath, len(paths))
	for index := range paths {
		copied[index] = append(schemaOptionalPath(nil), paths[index]...)
	}
	s.schemaRestorations[targetName] = schemaRestoration{
		paths: copied, originalHash: sha256.Sum256(original), normalizedHash: sha256.Sum256(normalized),
	}
}

func (s *Session) schemaRestoration(targetName string) []schemaOptionalPath {
	if s == nil || targetName == "" {
		return nil
	}
	return s.schemaRestorations[targetName].paths
}

func (s *Session) syntheticName(sourceName string) string {
	return s.syntheticToolName(llm.ToolKindCustom, sourceName)
}

func (s *Session) syntheticToolName(sourceKind llm.ToolKind, sourceName string) string {
	sourceKey := string(sourceKind) + "\x00" + sourceName
	if existing, ok := s.bySourceName[sourceKey]; ok {
		return existing
	}

	hasher := fnv.New32a()
	_, _ = hasher.Write([]byte(sourceName))
	base := fmt.Sprintf("axc_%08x", hasher.Sum32())
	name := base
	for suffix := 1; ; suffix++ {
		if _, occupied := s.occupiedNames[name]; !occupied {
			break
		}
		name = fmt.Sprintf("%s_%d", base, suffix)
	}
	s.occupiedNames[name] = struct{}{}
	s.bySourceName[sourceKey] = name
	s.bySyntheticName[name] = toolIdentity{SourceKind: sourceKind, SourceName: sourceName}
	return name
}

func (s *Session) identity(syntheticName string) (toolIdentity, bool) {
	if s == nil {
		return toolIdentity{}, false
	}
	identity, ok := s.bySyntheticName[syntheticName]
	return identity, ok
}

func (s *Session) addRestoreMiss() {
	if s == nil {
		return
	}
	s.restoreMiss.Add(1)
}

func (s *Session) addStreamViolation(err *llm.StreamInvariantError) {
	if s == nil || err == nil {
		return
	}
	s.streamViolations.Add(1)
	s.lastStreamViolation.Store(string(err.Code))
	s.lastStreamEvent.Store(string(err.EventKind))
}

func (s *Session) recordTerminalEvent(kind llm.EventKind) {
	if s == nil || !s.traceEnabled {
		return
	}
	switch kind {
	case llm.EventKindResponseCompleted, llm.EventKindResponseFailed,
		llm.EventKindResponseIncomplete, llm.EventKindResponseCancelled:
		s.terminalEvent.Store(string(kind))
	}
}

func (s *Session) setLowerNanos(nanos int64) {
	if s != nil {
		s.lowerNanos.Store(nanos)
	}
}

func (s *Session) addRestoreNanos(nanos int64) {
	if s != nil {
		s.restoreNanos.Add(nanos)
	}
}

func (s *Session) Summary() llm.ConversionTraceSummary {
	if s == nil || s.plan == nil {
		return llm.ConversionTraceSummary{}
	}
	summary := s.plan.Summary
	summary.LowerNanos = s.lowerNanos.Load()
	summary.RestoreNanos = s.restoreNanos.Load()
	summary.RestoreMiss = s.restoreMiss.Load()
	summary.CustomInputsRepaired = s.customInputsRepaired.Load()
	summary.IdentifiersNormalized = s.identifiersNormalized.Load()
	summary.SchemasNormalized = s.schemasNormalized.Load()
	summary.StreamViolations = s.streamViolations.Load()
	if value := s.lastStreamViolation.Load(); value != nil {
		summary.LastStreamViolation, _ = value.(string)
	}
	if value := s.lastStreamEvent.Load(); value != nil {
		if event, ok := value.(string); ok {
			summary.LastStreamEvent = llm.EventKind(event)
		}
	}
	if value := s.terminalEvent.Load(); value != nil {
		if event, ok := value.(string); ok {
			summary.TerminalEvent = llm.EventKind(event)
		}
	}
	if summary.RestoreMiss > 0 || summary.StreamViolations > 0 ||
		(summary.TerminalEvent != "" && summary.TerminalEvent != llm.EventKindResponseCompleted) {
		summary.Complete = false
	}
	return summary
}

func (s *Session) recordIdentifierNormalization(direction llm.ConversionDirection, ref ObjectRef) {
	if s == nil {
		return
	}
	s.identifiersNormalized.Add(1)
	s.recordDebug(direction, ref, "normalize", StrategyIdentifierNormalize, ReasonProtocolConstraint, true)
}

func (s *Session) recordSchemaNormalization(ref ObjectRef, reversible bool) {
	if s == nil {
		return
	}
	s.schemasNormalized.Add(1)
	s.recordDebug(llm.ConversionDirectionRequest, ref, "normalize", StrategySchemaNormalize, ReasonProtocolConstraint, reversible)
}

func (s *Session) recordCustomInputRepair(callID string, direction llm.ConversionDirection, ref ObjectRef) {
	if s == nil {
		return
	}
	if callID != "" {
		if _, loaded := s.customInputRepairSeen.LoadOrStore(callID, struct{}{}); loaded {
			return
		}
	}
	s.customInputsRepaired.Add(1)
	s.recordDebug(direction, ref, "repair", StrategyCustomAsFunction, ReasonSemanticProjection, true)
}

func (s *Session) recordCustomInputRawFallback(callID string) {
	if s == nil {
		return
	}
	if callID != "" {
		if _, loaded := s.customInputRawSeen.LoadOrStore(callID, struct{}{}); loaded {
			return
		}
	}
	s.addRestoreMiss()
}

func (s *Session) DebugTrace() *llm.ConversionDebugTrace {
	if s == nil || s.plan == nil {
		return nil
	}
	return s.plan.Debug.Clone()
}

func (s *Session) recordDebug(
	direction llm.ConversionDirection,
	ref ObjectRef,
	action string,
	strategy StrategyID,
	reason ReasonCode,
	reversible bool,
) {
	if s == nil || s.plan == nil || s.plan.Debug == nil {
		return
	}
	s.plan.Debug.Append(
		direction, string(ref.Kind), action, string(strategy), string(reason), reversible, objectRefBytes(ref),
	)
}
