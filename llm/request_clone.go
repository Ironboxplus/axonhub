package llm

import (
	"net/http"
	"reflect"

	"github.com/looplj/axonhub/llm/httpclient"
)

var (
	httpClientRequestPointerType = reflect.TypeFor[*httpclient.Request]()
	httpRequestPointerType       = reflect.TypeFor[*http.Request]()
)

// Clone returns an attempt-safe copy of the unified request.
//
// Outbound transformers and middleware are allowed to enrich request metadata
// or normalize request fields. A pipeline may execute more than one attempt,
// so sharing the same mutable request graph between attempts can leak a failed
// provider's mutations into the next provider. Clone isolates all mutable
// request data, including provider extensions and arbitrary transformer
// metadata.
//
// The net/http request referenced by httpclient.Request.RawRequest is retained
// by pointer. It carries the request context and transport internals and is
// treated as read-only by the pipeline. The surrounding httpclient.Request,
// including its body, headers, query, auth, and metadata, is cloned.
func (r *Request) Clone() *Request {
	if r == nil {
		return nil
	}

	cloned := cloneRequestValue(reflect.ValueOf(r), make(map[requestCloneVisit]reflect.Value))
	return cloned.Interface().(*Request)
}

type requestCloneVisit struct {
	typ  reflect.Type
	kind reflect.Kind
	ptr  uintptr
}

func cloneRequestValue(src reflect.Value, visited map[requestCloneVisit]reflect.Value) reflect.Value {
	if !src.IsValid() {
		return src
	}

	if src.Type() == httpClientRequestPointerType {
		if src.IsNil() {
			return reflect.Zero(src.Type())
		}
		return reflect.ValueOf(cloneHTTPClientRequest(src.Interface().(*httpclient.Request), visited))
	}
	if src.Type() == httpRequestPointerType {
		// A raw net/http request owns context and transport state. The pipeline
		// only reads it through the cloned httpclient.Request wrapper.
		return src
	}

	switch src.Kind() {
	case reflect.Interface:
		if src.IsNil() {
			return reflect.Zero(src.Type())
		}
		cloned := cloneRequestValue(src.Elem(), visited)
		dst := reflect.New(src.Type()).Elem()
		dst.Set(cloned)
		return dst

	case reflect.Pointer:
		if src.IsNil() {
			return reflect.Zero(src.Type())
		}
		visit := requestCloneVisit{typ: src.Type(), kind: src.Kind(), ptr: src.Pointer()}
		if cloned, ok := visited[visit]; ok {
			return cloned
		}
		dst := reflect.New(src.Type().Elem())
		visited[visit] = dst
		dst.Elem().Set(cloneRequestValue(src.Elem(), visited))
		return dst

	case reflect.Map:
		if src.IsNil() {
			return reflect.Zero(src.Type())
		}
		visit := requestCloneVisit{typ: src.Type(), kind: src.Kind(), ptr: src.Pointer()}
		if cloned, ok := visited[visit]; ok {
			return cloned
		}
		dst := reflect.MakeMapWithSize(src.Type(), src.Len())
		visited[visit] = dst
		iter := src.MapRange()
		for iter.Next() {
			// Map keys are identity-bearing values. Request metadata uses string
			// keys, and retaining other key identities is less surprising than
			// manufacturing new pointer keys.
			dst.SetMapIndex(iter.Key(), cloneRequestValue(iter.Value(), visited))
		}
		return dst

	case reflect.Slice:
		if src.IsNil() {
			return reflect.Zero(src.Type())
		}
		visit := requestCloneVisit{typ: src.Type(), kind: src.Kind(), ptr: src.Pointer()}
		if src.Pointer() != 0 {
			if cloned, ok := visited[visit]; ok {
				return cloned
			}
		}
		dst := reflect.MakeSlice(src.Type(), src.Len(), src.Len())
		if src.Pointer() != 0 {
			visited[visit] = dst
		}
		for i := 0; i < src.Len(); i++ {
			dst.Index(i).Set(cloneRequestValue(src.Index(i), visited))
		}
		return dst

	case reflect.Array:
		dst := reflect.New(src.Type()).Elem()
		for i := 0; i < src.Len(); i++ {
			dst.Index(i).Set(cloneRequestValue(src.Index(i), visited))
		}
		return dst

	case reflect.Struct:
		// Copy the whole value first so structs with unexported immutable
		// implementation fields remain intact, then recursively replace exported
		// request-model fields with isolated values.
		dst := reflect.New(src.Type()).Elem()
		dst.Set(src)
		for i := 0; i < src.NumField(); i++ {
			if src.Type().Field(i).PkgPath != "" || !dst.Field(i).CanSet() {
				continue
			}
			dst.Field(i).Set(cloneRequestValue(src.Field(i), visited))
		}
		return dst

	default:
		return src
	}
}

func cloneHTTPClientRequest(src *httpclient.Request, visited map[requestCloneVisit]reflect.Value) *httpclient.Request {
	if src == nil {
		return nil
	}

	visit := requestCloneVisit{
		typ:  httpClientRequestPointerType,
		kind: reflect.Pointer,
		ptr:  reflect.ValueOf(src).Pointer(),
	}
	if cloned, ok := visited[visit]; ok {
		return cloned.Interface().(*httpclient.Request)
	}

	dst := *src
	clonedRequest := &dst
	// Register the wrapper before cloning metadata so a provider extension can
	// safely point back to the request without causing infinite recursion.
	visited[visit] = reflect.ValueOf(clonedRequest)
	if src.Query != nil {
		dst.Query = make(map[string][]string, len(src.Query))
		for key, values := range src.Query {
			dst.Query[key] = append([]string(nil), values...)
		}
	}
	if src.Headers != nil {
		dst.Headers = src.Headers.Clone()
	}
	dst.Body = append([]byte(nil), src.Body...)
	dst.JSONBody = append([]byte(nil), src.JSONBody...)
	if src.Auth != nil {
		auth := *src.Auth
		dst.Auth = &auth
	}
	if src.Metadata != nil {
		dst.Metadata = cloneRequestValue(reflect.ValueOf(src.Metadata), visited).Interface().(map[string]string)
	}
	if src.TransformerMetadata != nil {
		dst.TransformerMetadata = cloneRequestValue(reflect.ValueOf(src.TransformerMetadata), visited).Interface().(map[string]any)
	}
	// See Clone: raw transport/context state is immutable and intentionally shared.
	dst.RawRequest = src.RawRequest
	return clonedRequest
}
