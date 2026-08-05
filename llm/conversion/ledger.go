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

type toolWireIdentityKey struct {
	name      string
	namespace string
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
	byWireIdentity        map[toolWireIdentityKey]toolIdentity
	occupiedNames         map[string]struct{}
	targetToolNames       map[string]*identifierLedger
	targetCallIDs         *identifierLedger
	sourceCallIDs         *identifierLedger
	schemaRestorations    map[toolWireIdentityKey]schemaRestoration
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
	debugMu               sync.Mutex
	compactEmulation      *compactEmulationState
}

func newSession(plan *Plan, request *llm.Request, traceEnabled bool) *Session {
	session := &Session{
		plan:                  plan,
		bySourceName:          make(map[string]string),
		bySyntheticName:       make(map[string]toolIdentity),
		byWireIdentity:        make(map[toolWireIdentityKey]toolIdentity),
		occupiedNames:         make(map[string]struct{}),
		schemaRestorations:    make(map[toolWireIdentityKey]schemaRestoration),
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
			identity := toolIdentity{
				SourceKind: llm.ToolKindFunction, SourceName: sourceName, SourceNamespace: definition.Function.Namespace,
			}
			session.registerToolIdentity(definition.LogicalName, definition.Function.Namespace, identity)
		}
		for _, tool := range request.Tools {
			if tool.Type == llm.ToolTypeFunction && tool.Function.Name != "" {
				session.occupiedNames[tool.Function.Name] = struct{}{}
			}
		}
	}
	return session
}

func (s *Session) registerSchemaRestoration(targetName, namespace string, paths []schemaOptionalPath, original, normalized []byte) {
	if s == nil || targetName == "" {
		return
	}
	if s.schemaRestorations == nil {
		s.schemaRestorations = make(map[toolWireIdentityKey]schemaRestoration)
	}
	copied := make([]schemaOptionalPath, len(paths))
	for index := range paths {
		copied[index] = append(schemaOptionalPath(nil), paths[index]...)
	}
	restoration := schemaRestoration{
		paths: copied, originalHash: sha256.Sum256(original), normalizedHash: sha256.Sum256(normalized),
	}
	s.schemaRestorations[toolWireIdentityKey{name: targetName}] = restoration
	if namespace != "" {
		wireName := strings.TrimPrefix(targetName, namespace+"__")
		s.schemaRestorations[toolWireIdentityKey{name: wireName, namespace: namespace}] = restoration
	}
}

func (s *Session) schemaRestoration(targetName, namespace string) []schemaOptionalPath {
	if s == nil || targetName == "" {
		return nil
	}
	if namespace != "" {
		if restoration, ok := s.schemaRestorations[toolWireIdentityKey{name: targetName, namespace: namespace}]; ok {
			return restoration.paths
		}
	}
	return s.schemaRestorations[toolWireIdentityKey{name: targetName}].paths
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
	s.registerToolIdentity(name, "", toolIdentity{SourceKind: sourceKind, SourceName: sourceName})
	return name
}

func (s *Session) registerToolIdentity(targetName, namespace string, identity toolIdentity) {
	if s == nil || targetName == "" {
		return
	}
	if s.bySyntheticName == nil {
		s.bySyntheticName = make(map[string]toolIdentity)
	}
	if s.byWireIdentity == nil {
		s.byWireIdentity = make(map[toolWireIdentityKey]toolIdentity)
	}
	s.bySyntheticName[targetName] = identity
	if namespace != "" {
		wireName := strings.TrimPrefix(targetName, namespace+"__")
		s.byWireIdentity[toolWireIdentityKey{name: wireName, namespace: namespace}] = identity
	}
}

func (s *Session) identity(targetName, namespace string) (toolIdentity, bool) {
	if s == nil {
		return toolIdentity{}, false
	}
	if namespace != "" {
		if identity, ok := s.byWireIdentity[toolWireIdentityKey{name: targetName, namespace: namespace}]; ok {
			return identity, true
		}
	}
	identity, ok := s.bySyntheticName[targetName]
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

func (s *Session) recordCustomInputRawFallback(callID string, direction llm.ConversionDirection, ref ObjectRef) {
	if s == nil {
		return
	}
	if callID != "" {
		if _, loaded := s.customInputRawSeen.LoadOrStore(callID, struct{}{}); loaded {
			return
		}
	}
	s.addRestoreMiss()
	s.recordDebug(direction, ref, "restore_miss", StrategyCustomAsFunction, ReasonNoStrategy, false)
}

func (s *Session) DebugTrace() *llm.ConversionDebugTrace {
	if s == nil || s.plan == nil {
		return nil
	}
	s.debugMu.Lock()
	trace := s.plan.Debug
	s.debugMu.Unlock()
	return trace.Clone()
}

func (s *Session) recordDebug(
	direction llm.ConversionDirection,
	ref ObjectRef,
	action string,
	strategy StrategyID,
	reason ReasonCode,
	reversible bool,
) {
	if s == nil || s.plan == nil {
		return
	}
	evidence := runtimeActionEvidence(direction, ref, action, strategy, reason, reversible)
	s.debugMu.Lock()
	trace := s.plan.Debug
	if trace == nil && requiredRuntimeEvidence(evidence) {
		trace = llm.NewRequiredConversionDebugTrace(1)
		s.plan.Debug = trace
	}
	s.debugMu.Unlock()
	if trace != nil {
		trace.Append(evidence, objectRefBytes(ref))
	}
}
