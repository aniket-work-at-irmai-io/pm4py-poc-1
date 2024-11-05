class FMEA:
    def __init__(self, failure_modes):
        self.failure_modes = failure_modes
        self.severity_scale = {1: "No effect", 5: "Moderate effect", 10: "Hazardous effect"}
        self.likelihood_scale = {1: "Very unlikely", 5: "Occasional", 10: "Inevitable"}
        self.detectability_scale = {1: "Certain detection", 5: "Moderate detection", 10: "No detection"}

    def assess_risk(self):
        results = []
        for mode in self.failure_modes:
            severity = self._get_user_input("Severity", self.severity_scale)
            likelihood = self._get_user_input("Likelihood", self.likelihood_scale)
            detectability = self._get_user_input("Detectability", self.detectability_scale)
            rpn = severity * likelihood * detectability
            results.append({
                "failure_mode": mode,
                "severity": severity,
                "likelihood": likelihood,
                "detectability": detectability,
                "rpn": rpn
            })
        return results



    def _get_user_input(self, factor, scale):
        print(f"\nAssess {factor} for failure mode:")
        for key, value in scale.items():
            print(f"{key}: {value}")
        return int(input(f"Enter {factor} score: "))