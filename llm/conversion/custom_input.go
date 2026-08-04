package conversion

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/looplj/axonhub/llm"
)

// customInputStatus distinguishes a provider-native envelope from the one
// bounded repair pass and the byte-exact fallback required for free-form
// tools. It deliberately carries no payload and is safe to aggregate.
type customInputStatus uint8

const (
	customInputExact customInputStatus = iota
	customInputRepaired
	customInputRaw
)

func decodeCustomInput(arguments string) (string, customInputStatus) {
	if input, ok := parseCustomInputEnvelope(arguments); ok {
		return input, customInputExact
	}

	// A model may double-encode the function envelope or return the requested
	// free-form value as a JSON string. Both forms have one unambiguous decode.
	var encoded string
	if json.Unmarshal([]byte(arguments), &encoded) == nil {
		if input, ok := parseCustomInputEnvelope(encoded); ok {
			return input, customInputRepaired
		}
		return encoded, customInputRepaired
	}

	candidate, fenced := unwrapSingleJSONFence(arguments)
	if fenced {
		if input, ok := parseCustomInputEnvelope(candidate); ok {
			return input, customInputRepaired
		}
	}
	if repaired, ok := repairCustomJSONContainer(candidate); ok {
		if input, valid := parseCustomInputEnvelope(repaired); valid {
			return input, customInputRepaired
		}
	}

	// There is no lossy error path for custom tools. If the deterministic
	// envelope repair cannot prove an input value, preserve every original byte
	// as the free-form call input so the source client still receives the call.
	return arguments, customInputRaw
}

func restoreCustomInput(
	arguments string,
	callID string,
	session *Session,
	direction llm.ConversionDirection,
	ref ObjectRef,
) (string, bool) {
	input, status := decodeCustomInput(arguments)
	switch status {
	case customInputExact:
		return input, true
	case customInputRepaired:
		session.recordCustomInputRepair(callID, direction, ref)
		return input, true
	default:
		session.recordCustomInputRawFallback(callID, direction, ref)
		return input, false
	}
}

func parseCustomInputEnvelope(arguments string) (string, bool) {
	var payload struct {
		Input *string `json:"input"`
	}
	if err := json.Unmarshal([]byte(arguments), &payload); err != nil || payload.Input == nil {
		return "", false
	}
	return *payload.Input, true
}

func unwrapSingleJSONFence(arguments string) (string, bool) {
	trimmed := strings.TrimSpace(arguments)
	if !strings.HasPrefix(trimmed, "```") || !strings.HasSuffix(trimmed, "```") {
		return strings.TrimSpace(arguments), false
	}
	lineEnd := strings.IndexByte(trimmed, '\n')
	if lineEnd < 0 {
		return strings.TrimSpace(arguments), false
	}
	language := strings.TrimSpace(trimmed[3:lineEnd])
	if language != "" && !strings.EqualFold(language, "json") {
		return strings.TrimSpace(arguments), false
	}
	body := strings.TrimSpace(trimmed[lineEnd+1 : len(trimmed)-3])
	return body, true
}

// repairCustomJSONContainer performs one linear, deterministic repair pass on
// a JSON object/array. It only escapes literal control bytes inside strings,
// removes commas immediately before a closing delimiter, and closes an
// otherwise well-nested truncated value. It never guesses keys, quotes bare
// text, rewrites escapes, or changes an already valid non-envelope value.
func repairCustomJSONContainer(arguments string) (string, bool) {
	candidate := strings.TrimSpace(arguments)
	if candidate == "" || candidate[0] != '{' && candidate[0] != '[' {
		return "", false
	}

	var output bytes.Buffer
	output.Grow(len(candidate) + 8)
	stack := make([]byte, 0, 8)
	inString := false
	escaped := false
	changed := candidate != arguments

	for index := 0; index < len(candidate); index++ {
		current := candidate[index]
		if inString {
			if escaped {
				output.WriteByte(current)
				escaped = false
				continue
			}
			switch current {
			case '\\':
				output.WriteByte(current)
				escaped = true
			case '"':
				output.WriteByte(current)
				inString = false
			case '\n':
				output.WriteString(`\n`)
				changed = true
			case '\r':
				output.WriteString(`\r`)
				changed = true
			case '\t':
				output.WriteString(`\t`)
				changed = true
			default:
				if current < 0x20 {
					const hex = "0123456789abcdef"
					output.WriteString(`\u00`)
					output.WriteByte(hex[current>>4])
					output.WriteByte(hex[current&0x0f])
					changed = true
				} else {
					output.WriteByte(current)
				}
			}
			continue
		}

		switch current {
		case '"':
			inString = true
			output.WriteByte(current)
		case '{', '[':
			stack = append(stack, current)
			output.WriteByte(current)
		case '}', ']':
			if len(stack) == 0 || !matchingJSONDelimiter(stack[len(stack)-1], current) {
				return "", false
			}
			stack = stack[:len(stack)-1]
			if removeTrailingJSONComma(&output) {
				changed = true
			}
			output.WriteByte(current)
		default:
			output.WriteByte(current)
		}
	}
	if escaped {
		return "", false
	}
	if inString {
		output.WriteByte('"')
		changed = true
	}
	for index := len(stack) - 1; index >= 0; index-- {
		if stack[index] == '{' {
			output.WriteByte('}')
		} else {
			output.WriteByte(']')
		}
		changed = true
	}
	if !changed || !json.Valid(output.Bytes()) {
		return "", false
	}
	return output.String(), true
}

func matchingJSONDelimiter(open, close byte) bool {
	return open == '{' && close == '}' || open == '[' && close == ']'
}

func removeTrailingJSONComma(output *bytes.Buffer) bool {
	if output == nil || output.Len() == 0 {
		return false
	}
	raw := output.Bytes()
	index := len(raw) - 1
	for index >= 0 && (raw[index] == ' ' || raw[index] == '\t' || raw[index] == '\r' || raw[index] == '\n') {
		index--
	}
	if index < 0 || raw[index] != ',' {
		return false
	}
	tail := append([]byte(nil), raw[index+1:]...)
	output.Truncate(index)
	_, _ = output.Write(tail)
	return true
}
