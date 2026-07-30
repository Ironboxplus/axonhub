package emulation

import (
	"fmt"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/emulation/hosted"
)

// CapabilityProfile returns the target protocol profile augmented only with
// emulators that are actually configured on this controller. It is safe for a
// router to cache because it contains no request IDs, tool names, or secrets.
func (controller *Controller) CapabilityProfile(target llm.APIFormat) (conversion.CapabilityProfile, bool) {
	profile, ok := conversion.ProfileFor(target)
	if !ok {
		return conversion.CapabilityProfile{}, false
	}
	if controller == nil {
		return profile, true
	}
	if controllerMCPConfigured(controller.config) {
		profile.EmulatedTools |= conversion.CapabilityMCPTool
	}
	if len(controller.config.Hosted.SyntheticNameKey) >= 32 {
		profile.EmulatedTools |= conversion.CapabilityToolSearch
		for kind, executor := range controller.config.Hosted.Executors {
			if executor == nil || executor.Kind() != kind {
				continue
			}
			profile.EmulatedTools |= conversion.CapabilityForToolKind(kind)
		}
	}
	if profile.EmulatedTools != 0 {
		profile.ID = fmt.Sprintf("%s+gateway-%x", profile.ID, uint32(profile.EmulatedTools))
	}
	return profile, true
}

// Preflight evaluates a candidate using protocol defaults plus the exact MCP
// and hosted executors registered on this controller. An incomplete result is
// intended to exclude the candidate before any provider request is emitted.
func (controller *Controller) Preflight(request *llm.Request, target llm.APIFormat) (*conversion.Plan, error) {
	profile, ok := controller.CapabilityProfile(target)
	if !ok {
		return conversion.NewPlanner().Plan(request, target)
	}
	return conversion.NewPlannerWithProfile(profile).Plan(request, target)
}

func controllerMCPConfigured(config ControllerConfig) bool {
	return (config.MCP.EndpointPolicy != nil || config.MCP.EndpointPolicyFactory != nil) && len(config.MCP.SyntheticNameKey) >= 32
}

func validateHostedConfig(config hosted.Config) error {
	if len(config.Executors) == 0 {
		return nil
	}
	if len(config.SyntheticNameKey) < 32 {
		return fmt.Errorf("%w: hosted executor synthetic name key is missing", hosted.ErrConfig)
	}
	for kind, executor := range config.Executors {
		if executor == nil {
			return fmt.Errorf("%w: hosted %s executor is nil", hosted.ErrConfig, kind)
		}
		if executor.Kind() != kind {
			return fmt.Errorf("%w: hosted executor kind %s is registered as %s", hosted.ErrConfig, executor.Kind(), kind)
		}
		if conversion.CapabilityForToolKind(kind) == 0 {
			return fmt.Errorf("%w: hosted executor kind %s is not planned", hosted.ErrConfig, kind)
		}
	}
	return nil
}
