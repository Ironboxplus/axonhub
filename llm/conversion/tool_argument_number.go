package conversion

import "encoding/json"

// maxCanonicalJSONIntegerDigits bounds expansion of one exponent token. Tool
// arguments are provider-controlled input: a value such as 1e999999999 must
// remain opaque instead of allocating an unbounded replacement.
const maxCanonicalJSONIntegerDigits = 64 << 10

// ToolArgumentNormalizationPolicy is the narrow deployment-policy extension
// point for future aggregate payload/stream limits. This generic conversion
// layer deliberately does not impose a new resource-governance contract: a
// configured gateway policy must decide whether a valid opaque argument is
// rejected, streamed, or retained.
// TODO: thread this policy through the gateway only when that contract exists.
type ToolArgumentNormalizationPolicy interface {
	PermitToolArgumentRewrite(inputBytes, rewrittenBytes int) bool
}

// canonicalizeIntegralJSONNumbers rewrites only JSON number tokens outside
// strings whose mathematical value is an integer. It deliberately operates on
// lexical tokens rather than decoding into float64, preserving large integers,
// object key order, whitespace, string content, and all non-integral numbers.
// Invalid JSON remains byte-for-byte opaque under the existing fail-closed
// contract.
func canonicalizeIntegralJSONNumbers(raw []byte) (json.RawMessage, bool) {
	if len(raw) == 0 || !json.Valid(raw) {
		return raw, false
	}

	var output []byte
	copyFrom := 0
	inString, escaped := false, false
	for index := 0; index < len(raw); {
		char := raw[index]
		if inString {
			if escaped {
				escaped = false
			} else if char == '\\' {
				escaped = true
			} else if char == '"' {
				inString = false
			}
			index++
			continue
		}
		if char == '"' {
			inString = true
			index++
			continue
		}
		if char != '-' && (char < '0' || char > '9') {
			index++
			continue
		}

		end := index + 1
		for end < len(raw) && isJSONNumberByte(raw[end]) {
			end++
		}
		if replacement, changed := canonicalJSONIntegerToken(raw[index:end]); changed {
			if output == nil {
				output = make([]byte, 0, len(raw))
			}
			output = append(output, raw[copyFrom:index]...)
			output = append(output, replacement...)
			copyFrom = end
		}
		index = end
	}
	if output == nil {
		return raw, false
	}
	output = append(output, raw[copyFrom:]...)
	return json.RawMessage(output), true
}

func isJSONNumberByte(char byte) bool {
	return char >= '0' && char <= '9' || char == '-' || char == '+' || char == '.' || char == 'e' || char == 'E'
}

// canonicalJSONIntegerToken returns an integer spelling when token has an
// exact integer value. json.Valid has already validated the outer document,
// but this parser is defensive and leaves an unexpected token untouched.
func canonicalJSONIntegerToken(token []byte) ([]byte, bool) {
	if len(token) == 0 {
		return nil, false
	}
	index := 0
	negative := false
	if token[index] == '-' {
		negative = true
		index++
		if index == len(token) {
			return nil, false
		}
	}
	integerStart := index
	for index < len(token) && token[index] >= '0' && token[index] <= '9' {
		index++
	}
	if index == integerStart {
		return nil, false
	}
	integer := token[integerStart:index]
	fraction := []byte(nil)
	if index < len(token) && token[index] == '.' {
		index++
		fractionStart := index
		for index < len(token) && token[index] >= '0' && token[index] <= '9' {
			index++
		}
		if index == fractionStart {
			return nil, false
		}
		fraction = token[fractionStart:index]
	}
	exponent := int64(0)
	if index < len(token) && (token[index] == 'e' || token[index] == 'E') {
		index++
		exponentNegative := false
		if index < len(token) && (token[index] == '+' || token[index] == '-') {
			exponentNegative = token[index] == '-'
			index++
		}
		exponentStart := index
		for index < len(token) && token[index] >= '0' && token[index] <= '9' {
			index++
		}
		if index == exponentStart || index != len(token) {
			return nil, false
		}
		var bounded bool
		exponent, bounded = boundedDecimalExponent(token[exponentStart:index], exponentNegative)
		if !bounded {
			// A nonzero value with a huge positive exponent cannot fit within the
			// bounded replacement; a huge negative exponent cannot be integral.
			// The all-zero case is handled below without using the exponent.
			exponent = 0
			if !allZero(integer, fraction) {
				return nil, false
			}
		}
	} else if index != len(token) {
		return nil, false
	}

	if allZero(integer, fraction) {
		return []byte("0"), string(token) != "0"
	}

	digits := make([]byte, 0, len(integer)+len(fraction))
	digits = append(digits, integer...)
	digits = append(digits, fraction...)
	firstNonZero := 0
	for firstNonZero < len(digits) && digits[firstNonZero] == '0' {
		firstNonZero++
	}
	digits = digits[firstNonZero:]
	scale := exponent - int64(len(fraction))
	if scale < 0 {
		needZeros := -scale
		if needZeros > int64(len(digits)) || !hasTrailingZeros(digits, int(needZeros)) {
			return nil, false
		}
		digits = digits[:len(digits)-int(needZeros)]
	} else if scale > 0 {
		if scale > maxCanonicalJSONIntegerDigits || int64(len(digits))+scale > maxCanonicalJSONIntegerDigits {
			return nil, false
		}
		for count := int64(0); count < scale; count++ {
			digits = append(digits, '0')
		}
	}
	if negative {
		replacement := make([]byte, 0, len(digits)+1)
		replacement = append(replacement, '-')
		replacement = append(replacement, digits...)
		if string(replacement) == string(token) {
			return nil, false
		}
		return replacement, true
	}
	if string(digits) == string(token) {
		return nil, false
	}
	return digits, true
}

func boundedDecimalExponent(raw []byte, negative bool) (int64, bool) {
	var value int64
	for _, char := range raw {
		if value > maxCanonicalJSONIntegerDigits/10 {
			return 0, false
		}
		value = value*10 + int64(char-'0')
		if value > maxCanonicalJSONIntegerDigits {
			return 0, false
		}
	}
	if negative {
		value = -value
	}
	return value, true
}

func allZero(parts ...[]byte) bool {
	for _, part := range parts {
		for _, char := range part {
			if char != '0' {
				return false
			}
		}
	}
	return true
}

func hasTrailingZeros(value []byte, count int) bool {
	if count > len(value) {
		return false
	}
	for _, char := range value[len(value)-count:] {
		if char != '0' {
			return false
		}
	}
	return true
}
