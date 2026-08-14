package conversion

import (
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestCanonicalizeIntegralJSONNumbersPreservesLexicalSafety(t *testing.T) {
	t.Parallel()

	const raw = `{"timeout_ms":1000.0,"scientific":1e3,"negative_zero":-0.0,"scaled":1000e-3,"nested":{"array":[1000.0,1e3,-0.0,9007199254740993.0]},"fraction":12.5,"tiny":1e-3,"string":"1000.0","1000.0":"key","escaped":"\\\"1000.0\\\""}`
	const want = `{"timeout_ms":1000,"scientific":1000,"negative_zero":0,"scaled":1,"nested":{"array":[1000,1000,0,9007199254740993]},"fraction":12.5,"tiny":1e-3,"string":"1000.0","1000.0":"key","escaped":"\\\"1000.0\\\""}`

	got, changed := canonicalizeIntegralJSONNumbers([]byte(raw))
	if !changed {
		t.Fatal("expected integral numeric lexemes to be canonicalized")
	}
	if string(got) != want {
		t.Fatalf("canonicalized arguments = %s\nwant = %s", got, want)
	}
}

func TestCanonicalizeIntegralJSONNumbersNeverUsesFloatPrecisionOrRepairsInvalidJSON(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		raw     string
		want    string
		changed bool
	}{
		{name: "large integer is exact", raw: `{"n":9007199254740993.0}`, want: `{"n":9007199254740993}`, changed: true},
		{name: "large scaled integer is exact", raw: `{"n":123456789012345678901234567890e2}`, want: `{"n":12345678901234567890123456789000}`, changed: true},
		{name: "non integral remains lexical", raw: `{"n":123456789012345678901234567890.25,"small":1e-3}`, want: `{"n":123456789012345678901234567890.25,"small":1e-3}`, changed: false},
		{name: "invalid json remains opaque", raw: `{"n":1000.0`, want: `{"n":1000.0`, changed: false},
		{name: "zero huge exponent is bounded", raw: `{"n":-0e999999999}`, want: `{"n":0}`, changed: true},
		{name: "huge materialized integer stays original", raw: `{"n":1e999999999}`, want: `{"n":1e999999999}`, changed: false},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, changed := canonicalizeIntegralJSONNumbers([]byte(test.raw))
			if changed != test.changed || string(got) != test.want {
				t.Fatalf("canonicalize(%s) = (%s, %v), want (%s, %v)", test.raw, got, changed, test.want, test.changed)
			}
		})
	}
}

func TestCanonicalizeIntegralJSONNumbersPreservesWhitespaceAndKeyOrder(t *testing.T) {
	t.Parallel()
	const raw = "{\n  \"z\" : 1000.0 ,\n  \"first\" : 1e3, \"fraction\" : 1.25,\n  \"literal\" : \"1e3\"\n}"
	const want = "{\n  \"z\" : 1000 ,\n  \"first\" : 1000, \"fraction\" : 1.25,\n  \"literal\" : \"1e3\"\n}"

	got, changed := canonicalizeIntegralJSONNumbers([]byte(raw))
	if !changed || string(got) != want {
		t.Fatalf("whitespace/key-order preserving canonicalization = %q, changed=%v\nwant %q", got, changed, want)
	}
}

func TestStripOptionalNullArgumentsKeepsLargeJSONNumbersExact(t *testing.T) {
	t.Parallel()
	const raw = `{"before":9007199254740993.0,"optional":null,"after":123456789012345678901234567890}`
	got, changed := stripOptionalNullArguments([]byte(raw), []schemaOptionalPath{{{property: "optional"}}})
	if !changed {
		t.Fatal("expected optional null field to be restored away")
	}
	const want = `{"after":123456789012345678901234567890,"before":9007199254740993.0}`
	if string(got) != want {
		t.Fatalf("optional-null restoration changed an exact number: %s\nwant %s", got, want)
	}
}

func TestCanonicalJSONIntegerTokenDefensiveAndBoundaryBranches(t *testing.T) {
	t.Parallel()
	tests := []struct {
		token   string
		want    string
		changed bool
	}{
		{token: "", want: "", changed: false},
		{token: "-", want: "", changed: false},
		{token: ".1", want: "", changed: false},
		{token: "1.", want: "", changed: false},
		{token: "1e", want: "", changed: false},
		{token: "1x", want: "", changed: false},
		{token: "1000", want: "", changed: false},
		{token: "-1", want: "", changed: false},
		{token: "0001.0", want: "1", changed: true},
		{token: "-1.0", want: "-1", changed: true},
		{token: "1e-2", want: "", changed: false},
		{token: "12e-1", want: "", changed: false},
		{token: "1e65537", want: "", changed: false},
		{token: "1e99999", want: "", changed: false},
		{token: "0e99999", want: "0", changed: true},
	}
	for _, test := range tests {
		test := test
		t.Run(test.token, func(t *testing.T) {
			got, changed := canonicalJSONIntegerToken([]byte(test.token))
			if string(got) != test.want || changed != test.changed {
				t.Fatalf("canonicalJSONIntegerToken(%q) = (%q, %v), want (%q, %v)", test.token, got, changed, test.want, test.changed)
			}
		})
	}
	if _, bounded := boundedDecimalExponent([]byte(""), false); !bounded {
		t.Fatal("empty defensive exponent should remain bounded")
	}
	if _, bounded := boundedDecimalExponent([]byte("65537"), false); bounded {
		t.Fatal("over-limit exponent must be bounded")
	}
	if _, bounded := boundedDecimalExponent([]byte("999999"), true); bounded {
		t.Fatal("early overflowing exponent must be bounded")
	}
	if hasTrailingZeros([]byte("0"), 2) || hasTrailingZeros([]byte("120"), 2) || !hasTrailingZeros([]byte("1200"), 2) {
		t.Fatal("trailing-zero classifier boundary is incorrect")
	}
	if got, changed := canonicalJSONIntegerToken([]byte(strings.Repeat("9", maxCanonicalJSONIntegerDigits) + "e1")); got != nil || changed {
		t.Fatalf("oversized materialized integer must remain opaque: (%q, %v)", got, changed)
	}
}

func TestToolArgumentCanonicalizationLedgerIsBoundedAndDeduplicated(t *testing.T) {
	t.Parallel()
	var nilSession *Session
	nilSession.recordToolArgumentCanonicalization("response", ObjectRef{}, "call_nil")

	session := &Session{plan: &Plan{Summary: llm.ConversionTraceSummary{Complete: true}}}
	first := ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: 0, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
	second := ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: 1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
	session.recordToolArgumentCanonicalization("response", first, "call_same")
	session.recordToolArgumentCanonicalization("response", second, "call_same")
	session.recordToolArgumentCanonicalization("response", second, "")
	if got := session.Summary().ToolArgumentsCanonicalized; got != 2 {
		t.Fatalf("bounded canonicalization summary = %d, want 2", got)
	}
	trace := session.DebugTrace()
	if trace == nil || len(trace.Actions) != 2 || trace.Actions[0].FieldPath != "output[0].tool_call.arguments" || trace.Actions[1].FieldPath != "output[1].tool_call.arguments" {
		t.Fatalf("canonicalization evidence = %#v", trace)
	}
}

func BenchmarkCanonicalizeIntegralJSONNumbers(b *testing.B) {
	raw := []byte(`{"timeout_ms":1000.0,"scientific":1e3,"nested":[9007199254740993.0,{"fraction":1.25}],"literal":"1000.0"}`)
	b.ReportAllocs()
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		if _, changed := canonicalizeIntegralJSONNumbers(raw); !changed {
			b.Fatal("expected numeric fixture to be canonicalized")
		}
	}
}
