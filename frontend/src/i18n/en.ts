/**
 * Centralized English copy for VoiceGuard interface.
 *
 * Enforces strict honesty constraints per 09 §10:
 * - Never claims 100% detection, guarantees, or definitive proof.
 * - Uses calibrated risk language: "indicators consistent with", "estimated risk",
 *   "the model attended to", "we recommend verifying independently".
 */

export const COPY = {
  app: {
    title: 'VoiceGuard',
    subtitle: 'Multi-Signal Deepfake Voice & Scam Intelligence',
    description:
      'An automated assessment combining acoustic deepfake detection, multilingual scam-intent analysis, and interactive challenge verification.',
  },
  verdicts: {
    LOW: {
      label: 'Low Risk',
      summary: 'No strong indicators of synthetic speech or scam content were found.',
      guidance: 'This is not a guarantee of authenticity. Stay cautious with unexpected requests.',
    },
    MODERATE: {
      label: 'Moderate Risk',
      summary: 'Some indicators were detected, but the signals are not conclusive.',
      guidance: 'Verify the caller independently before acting on anything discussed.',
    },
    HIGH: {
      label: 'High Risk',
      summary: 'Multiple strong indicators of synthetic speech or scam content were detected.',
      guidance: 'Do not act on this call. Contact the person or organisation using a number you already trust.',
    },
    INCONCLUSIVE: {
      label: 'Inconclusive',
      summary: 'The audio could not be assessed reliably.',
      guidance: (reason: string) => `${reason}. Try a longer or clearer recording.`,
    },
  },
  disclaimers: {
    mandatoryScope:
      'VoiceGuard provides probabilistic assessments based on audio and transcript signals. It does not provide legal proof of identity or fabrication.',
    gradCam:
      'This visualisation shows which regions of the audio spectrogram most influenced the model\'s classification. It indicates where the model attended, not where manipulation definitively occurred. Attention maps on audio spectrograms can highlight recording artefacts, silence, or channel characteristics rather than synthesis artefacts.',
  },
} as const;
