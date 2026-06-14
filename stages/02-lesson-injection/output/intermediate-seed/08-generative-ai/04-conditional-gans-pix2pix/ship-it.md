## Ship It

Deploying Pix2Pix in production requires attention to three failure modes that don't appear in tutorial settings.

**Mode collapse on the condition.** If your paired dataset has limited diversity per condition, G learns to output the mean target for each input regardless of actual variation. The L1 term masks this initially — pixel-level error looks fine — but outputs look smeared. Diagnose by computing per-pixel variance across multiple runs of G on the same input with different dropout seeds. If variance is near zero, G is ignoring the stochastic channel.

**Discinator overpowering G.** PatchGAN converges faster than a full-image discriminator because each patch is an independent training signal. If D becomes too confident too early, G's gradient vanishes. Monitor the ratio of D accuracy to G loss. If D accuracy exceeds 0.95 for sustained steps, reduce D's learning rate by half or update G twice per D update.

**Domain mismatch at inference.** Pix2Pix assumes the test distribution matches training. If you train on daytime-to-nighttime translation and deploy on images with different lighting, the generator produces artifacts. There is no fix for this within Pix2Pix itself — you either retrain on the new domain or add a preprocessing step to normalize input statistics.

For the GTM parallel: when shipping a CRM enrichment pipeline with conditional logic, the same failure modes appear. Mode collapse maps to enrichment that always returns the same values regardless of input — a sign your data provider is returning cached or templated responses. Discriminator overpowering maps to over-filtering: your conditions are so strict that few records pass through, and you starve your sales team of pipeline. Domain mismatch maps to applying ICP filters trained on one segment (e.g., US SaaS) to a different segment (e.g., EU enterprise) without recalibration.

The production checklist: log the condition pass rate, log per-field enrichment success rate, and alert when either drops below your historical baseline. This is the CRM equivalent of monitoring G's loss curves and D's accuracy — you're tracking whether the conditioning signal is still doing its job.

[CITATION NEEDED — concept: Clay enrichment waterfall monitoring and condition pass rate tracking]