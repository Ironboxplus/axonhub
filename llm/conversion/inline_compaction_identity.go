package conversion

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
)

// digestInlineCompactionToken creates evidence and public IDs without
// retaining an opaque checkpoint value beyond the caller's operation.
func digestInlineCompactionToken(token string) string {
	digest := sha256.Sum256([]byte(token))
	return hex.EncodeToString(digest[:])
}

// inlineCompactionPublicIDs derives a stable correlated response/item pair
// from the opaque reference without exposing that reference to the client.
func inlineCompactionPublicIDs(token string) (responseID, itemID string) {
	digest := digestInlineCompactionToken(token)
	const suffixBytes = 24
	if len(digest) > suffixBytes {
		digest = digest[:suffixBytes]
	}
	return "resp_axcmp_" + digest, "cmp_axcmp_" + digest
}

func inlineCompactionCodecErrorCode(err error) InlineCompactionErrorCode {
	var failure CompactionStateCodecFailure
	if errors.As(err, &failure) && failure.CompactionStateCodecFailureCode() == CompactionStateCodecFailureInvalid {
		return InlineCompactionInvalidToken
	}
	return InlineCompactionCodecUnavailable
}
