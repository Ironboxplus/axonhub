//go:build !race

package responses

import "testing"

func TestResponsesIdentityResidualAllocationSlope(t *testing.T) {
	inbound := NewInboundTransformer()
	outbound, err := NewOutboundTransformer("https://example.invalid", "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	smallBody := nestedResidualRequestBody(64)
	largeBody := nestedResidualRequestBody(512)
	var runErr error
	small := testing.AllocsPerRun(3, func() {
		if runErr == nil {
			runErr = runResponsesIdentityResidual(smallBody, inbound, outbound)
		}
	})
	if runErr != nil {
		t.Fatal(runErr)
	}
	large := testing.AllocsPerRun(3, func() {
		if runErr == nil {
			runErr = runResponsesIdentityResidual(largeBody, inbound, outbound)
		}
	})
	if runErr != nil {
		t.Fatal(runErr)
	}

	if small > 14_500 {
		t.Fatalf("64-object Responses identity allocations = %.0f, budget 14500", small)
	}
	allocationsPerAdditionalObject := (large - small) / (512 - 64)
	if allocationsPerAdditionalObject > 215 {
		t.Fatalf("Responses identity allocation slope = %.1f/object, budget 215", allocationsPerAdditionalObject)
	}
}

func TestResponsesStreamIdentityResidualAllocationSlope(t *testing.T) {
	smallEvents := nestedResidualStreamEvents(32)
	largeEvents := nestedResidualStreamEvents(128)
	var runErr error
	small := testing.AllocsPerRun(3, func() {
		if runErr == nil {
			runErr = runResponsesStreamIdentityResidual(smallEvents)
		}
	})
	if runErr != nil {
		t.Fatal(runErr)
	}
	large := testing.AllocsPerRun(3, func() {
		if runErr == nil {
			runErr = runResponsesStreamIdentityResidual(largeEvents)
		}
	})
	if runErr != nil {
		t.Fatal(runErr)
	}

	if small > 5_200 {
		t.Fatalf("32-delta Responses stream allocations = %.0f, budget 5200", small)
	}
	allocationsPerAdditionalDelta := (large - small) / (128 - 32)
	if allocationsPerAdditionalDelta > 120 {
		t.Fatalf("Responses stream allocation slope = %.1f/delta, budget 120", allocationsPerAdditionalDelta)
	}
}
