package responses

import (
	"errors"
	"sync"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
)

// WebSocketExecutorRegistry owns Responses WebSocket session pools across
// short-lived transformer instances. Long-lived orchestrators commonly build
// one transformer per routing attempt; keeping the ownership here lets those
// attempts reuse the same upstream connection without making transformer
// instances global or retaining API keys in an application-side cache.
type WebSocketExecutorRegistry struct {
	mu        sync.Mutex
	executors map[any]*WebSocketExecutor
	closed    bool
}

func NewWebSocketExecutorRegistry() *WebSocketExecutorRegistry {
	return &WebSocketExecutorRegistry{executors: make(map[any]*WebSocketExecutor)}
}

func (registry *WebSocketExecutorRegistry) executorFor(inner pipeline.Executor) pipeline.Executor {
	if registry == nil {
		return NewWebSocketExecutor(inner)
	}
	identity, reusable := webSocketExecutorIdentity(inner)
	if !reusable {
		return NewWebSocketExecutor(inner)
	}

	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.closed {
		// Preserve request functionality during a concurrent graceful shutdown,
		// but do not repopulate the process-owned registry.
		return NewWebSocketExecutor(inner)
	}
	if executor := registry.executors[identity]; executor != nil {
		return executor
	}
	executor := NewWebSocketExecutor(inner)
	registry.executors[identity] = executor
	return executor
}

// Close releases every session pool owned by the registry. It is idempotent.
func (registry *WebSocketExecutorRegistry) Close() error {
	if registry == nil {
		return nil
	}
	registry.mu.Lock()
	if registry.closed {
		registry.mu.Unlock()
		return nil
	}
	registry.closed = true
	executors := make([]*WebSocketExecutor, 0, len(registry.executors))
	for identity, executor := range registry.executors {
		delete(registry.executors, identity)
		executors = append(executors, executor)
	}
	registry.mu.Unlock()

	errorsSeen := make([]error, 0, len(executors))
	for _, executor := range executors {
		if err := executor.Close(); err != nil {
			errorsSeen = append(errorsSeen, err)
		}
	}
	return errors.Join(errorsSeen...)
}

func webSocketExecutorIdentity(executor pipeline.Executor) (any, bool) {
	if client, ok := executor.(*httpclient.HttpClient); ok {
		if native := client.GetNativeClient(); native != nil {
			return native, true
		}
	}
	if ExecutorComparable(executor) {
		return executor, true
	}
	return nil, false
}
