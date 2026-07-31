package conversion

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

type ContinuationIdentity struct {
	CallID, ToolName, Namespace string
	Kind                        llm.ToolKind
}

// ContinuationBinding is injected once by the authenticated host for a chosen
// upstream attempt. The host owns trust, routing namespace, transactionality,
// persistence, TTL and replay policy.
type ContinuationBinding interface {
	Resolve(context.Context, ContinuationIdentity) ([]byte, bool, error)
	Capture(context.Context, ContinuationIdentity, []byte) error
}

type OutboundOption func(*Outbound)

func WithContinuationBinding(binding ContinuationBinding) OutboundOption {
	return func(outbound *Outbound) { outbound.continuation = binding }
}

func continuationIdentity(callID, name, namespace string, kind llm.ToolKind) ContinuationIdentity {
	return ContinuationIdentity{CallID: callID, ToolName: name, Namespace: namespace, Kind: kind}
}

// MemoryArgumentStore is a process-local, TTL-bounded map of provider-raw
// function_call.arguments bytes. Hosts scope entries with ContextScopedBinding
// so tenants never share keys. Values are raw provider bytes only — never logs.
type MemoryArgumentStore struct {
	mu      sync.Mutex
	entries map[string]memoryArgumentEntry
	ttl     time.Duration
	maxSize int
}

type memoryArgumentEntry struct {
	raw       []byte
	expiresAt time.Time
}

const (
	defaultArgumentContinuityTTL     = 2 * time.Hour
	defaultArgumentContinuityMaxSize = 8192
)

func NewMemoryArgumentStore() *MemoryArgumentStore {
	return &MemoryArgumentStore{
		entries: make(map[string]memoryArgumentEntry),
		ttl:     defaultArgumentContinuityTTL,
		maxSize: defaultArgumentContinuityMaxSize,
	}
}

func (s *MemoryArgumentStore) put(key string, raw []byte) {
	if s == nil || key == "" || len(raw) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.entries == nil {
		s.entries = make(map[string]memoryArgumentEntry)
	}
	now := time.Now()
	s.expireLocked(now)
	if len(s.entries) >= s.maxSize {
		s.evictOneLocked()
	}
	s.entries[key] = memoryArgumentEntry{
		raw:       append([]byte(nil), raw...),
		expiresAt: now.Add(s.ttl),
	}
}

func (s *MemoryArgumentStore) get(key string) ([]byte, bool) {
	if s == nil || key == "" {
		return nil, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.entries[key]
	if !ok {
		return nil, false
	}
	if time.Now().After(entry.expiresAt) {
		delete(s.entries, key)
		return nil, false
	}
	return append([]byte(nil), entry.raw...), true
}

func (s *MemoryArgumentStore) expireLocked(now time.Time) {
	for key, entry := range s.entries {
		if now.After(entry.expiresAt) {
			delete(s.entries, key)
		}
	}
}

func (s *MemoryArgumentStore) evictOneLocked() {
	var oldestKey string
	var oldest time.Time
	first := true
	for key, entry := range s.entries {
		if first || entry.expiresAt.Before(oldest) {
			oldestKey = key
			oldest = entry.expiresAt
			first = false
		}
	}
	if oldestKey != "" {
		delete(s.entries, oldestKey)
	}
}

// ContextScopedBinding stores and resolves raw arguments under the trusted
// SessionScope from context. Missing scope is a deliberate no-op so unscoped
// requests never cross-contaminate.
type ContextScopedBinding struct {
	store *MemoryArgumentStore
}

func NewContextScopedBinding(store *MemoryArgumentStore) *ContextScopedBinding {
	if store == nil {
		store = NewMemoryArgumentStore()
	}
	return &ContextScopedBinding{store: store}
}

// DefaultArgumentContinuity is shared by hosts that want process-local
// multi-turn argument replay without standing up external storage.
var DefaultArgumentContinuity = NewContextScopedBinding(NewMemoryArgumentStore())

func (b *ContextScopedBinding) Resolve(ctx context.Context, id ContinuationIdentity) ([]byte, bool, error) {
	if b == nil || b.store == nil {
		return nil, false, nil
	}
	key, ok := continuityStorageKey(ctx, id)
	if !ok {
		return nil, false, nil
	}
	raw, found := b.store.get(key)
	return raw, found, nil
}

func (b *ContextScopedBinding) Capture(ctx context.Context, id ContinuationIdentity, raw []byte) error {
	if b == nil || b.store == nil || len(raw) == 0 {
		return nil
	}
	key, ok := continuityStorageKey(ctx, id)
	if !ok {
		return nil
	}
	b.store.put(key, raw)
	return nil
}

func continuityStorageKey(ctx context.Context, id ContinuationIdentity) (string, bool) {
	if id.CallID == "" || id.ToolName == "" {
		return "", false
	}
	scope, ok := shared.GetSessionScope(ctx)
	if !ok || scope == "" {
		return "", false
	}
	// call_id is the primary correlation; tool name defends against accidental
	// id reuse across different tools inside one tenant scope.
	return scope + "\x00" + id.CallID + "\x00" + id.ToolName, true
}

func resolveContinuation(ctx context.Context, request *llm.Request, binding ContinuationBinding) {
	if request == nil || binding == nil {
		return
	}
	for index := range request.Input {
		call := request.Input[index].ToolCall
		if call == nil || call.Kind != llm.ToolKindFunction || call.CallID == "" || call.LogicalName == "" {
			continue
		}
		raw, ok, err := binding.Resolve(ctx, continuationIdentity(call.CallID, call.LogicalName, call.Namespace, call.Kind))
		if err == nil && ok && len(raw) > 0 {
			call.ArgumentsJSON, call.ArgumentsText = append(json.RawMessage(nil), raw...), ""
		}
	}
	for messageIndex := range request.Messages {
		message := &request.Messages[messageIndex]
		for callIndex := range message.ToolCalls {
			call := &message.ToolCalls[callIndex]
			if call.Type == llm.ToolTypeResponsesCustomTool || call.Function.Name == "" || call.ID == "" {
				continue
			}
			raw, ok, err := binding.Resolve(ctx, continuationIdentity(call.ID, call.Function.Name, call.Function.Namespace, llm.ToolKindFunction))
			if err == nil && ok && len(raw) > 0 {
				call.Function.Arguments = string(raw)
			}
		}
	}
}

func (s *Session) captureContinuation(ctx context.Context, id ContinuationIdentity, raw []byte) {
	if s == nil || s.continuation == nil || id.CallID == "" || id.ToolName == "" || len(raw) == 0 {
		return
	}
	_ = s.continuation.Capture(ctx, id, append([]byte(nil), raw...))
}

func (s *Session) recordProviderArgumentBytes(callID, name string, rawJSON []byte, rawText string) {
	if s == nil || s.plan == nil || s.plan.Target.APIFormat != llm.APIFormatOpenAIResponse || callID == "" {
		return
	}
	raw := rawJSON
	if len(raw) == 0 {
		raw = []byte(rawText)
	}
	if len(raw) == 0 {
		return
	}
	copied := append(json.RawMessage(nil), raw...)
	// Primary key is provider call id: names may be rewritten by identity
	// restore before publish, and call ids are unique within an attempt.
	s.providerArgumentBytes[providerArgumentRecordKey{callID: callID}] = copied
	if name != "" {
		s.providerArgumentBytes[providerArgumentRecordKey{callID: callID, name: name}] = copied
	}
}

func (s *Session) publishProviderArgumentBytes(ctx context.Context, providerCallID, clientCallID, name string) {
	if s == nil {
		return
	}
	raw := s.providerArgumentBytes[providerArgumentRecordKey{callID: providerCallID, name: name}]
	if len(raw) == 0 {
		raw = s.providerArgumentBytes[providerArgumentRecordKey{callID: providerCallID}]
	}
	if len(raw) == 0 {
		raw = s.providerArgumentBytes[providerArgumentRecordKey{callID: clientCallID, name: name}]
	}
	if len(raw) == 0 {
		raw = s.providerArgumentBytes[providerArgumentRecordKey{callID: clientCallID}]
	}
	if len(raw) == 0 {
		return
	}
	publishID := clientCallID
	if publishID == "" {
		publishID = providerCallID
	}
	if name == "" {
		return
	}
	s.captureContinuation(ctx, continuationIdentity(publishID, name, "", llm.ToolKindFunction), raw)
}
