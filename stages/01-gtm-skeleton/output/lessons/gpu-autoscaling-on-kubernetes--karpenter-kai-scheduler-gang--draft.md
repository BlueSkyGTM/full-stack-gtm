# GPU Autoscaling on Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling

## Beat 1: Hook

GPU idle time is a burn-rate multiplier. A cluster that takes 12 minutes to provision an A100 node for a 20-minute training run is spending 38% of its timeline waiting. This lesson covers the three mechanisms that close that gap: fast provisioning (Karpenter), GPU-aware queue scheduling (KAI Scheduler), and atomic pod-group admission (Gang Scheduling).

---

## Beat 2: Concept

**Node provisioning latency.** The default Kubernetes scheduler finds a node; it does not create one. Cluster Autoscaler polls every 10 seconds, then calls cloud APIs that take 5–10 minutes to return a GPU node. Karpenter inverts this: it watches for unschedulable pods directly, then calls instance APIs in parallel, achieving node-ready in 60–90 seconds for GPU types. Mechanism: event-driven provisioning loop replaces poll-and-wait.

**GPU-aware scheduling.** Default kube-scheduler bins packs on CPU/memory, ignoring GPU topology or fractional utilization. KAI Scheduler (originally Run:ai, now open-source) introduces queue-based scheduling with GPU fraction support, priority preemption, and fairness guarantees across teams. Mechanism: a scheduling queue with resource quotas and bin-packing heuristics tuned for GPU topology.

**Gang scheduling.** Distributed training jobs require all pods (e.g., 8 workers) to start simultaneously. If only 6 of 8 land, the job hangs and holds GPU memory indefinitely. Gang scheduling (implemented via `PodGroup` CRD in kube-scheduler plugins or Volcano) withholds admission until all pods in the group can be placed atomically. Mechanism: co-scheduling with all-or-nothing admission control.

**Interaction pattern.** These three layers compose: Karpenter provisions nodes fast, KAI Scheduler decides which jobs get which GPUs, and Gang Scheduling ensures multi-pod jobs start atomically or not at all.

---

## Beat 3: Demonstration

Show observable scheduling behavior using Kubernetes API mock outputs and timing data. No live cluster required — use `kubectl` dry-run and JSONPath to expose what the scheduler sees.

**Exercise hooks:**
- *(Easy)* Print the scheduling gates on a pending `PodGroup` CRD and identify which condition is blocking admission.
- *(Medium)* Given a 4-pod gang and a 2-node cluster with 1 GPU each, trace the scheduling decision and explain which pods land and which wait.
- *(Hard)* Simulate a priority preemption event: a high-priority job arrives, and KAI Scheduler must evict a low-priority job. Print the before/after queue state.

---

## Beat 4: Use It

**GTM Redirect:** Foundational for Zone 1 (AI Infrastructure). GPU scheduling directly controls unit economics of inference and training workloads. In GTM terms, a cluster that autoscales correctly means your per-inference cost stays predictable under load — the same mechanism that keeps training pipelines unblocked keeps customer-facing inference endpoints from over-provisioning.

**Exercise hooks:**
- *(Medium)* Calculate the cost impact: given a $3.50/hr A100 node, a 10-minute provisioning delay, and 50 jobs/day, compute monthly waste. Write the calculation as a shell one-liner that outputs the dollar amount.

---

## Beat 5: Ship It

Production deployment patterns for each layer. Karpenter `NodePool` configuration with GPU instance families and disruption budgets. KAI Scheduler installation via Helm with queue definitions mapped to team namespaces. PodGroup annotations on `Job` or `MPIJob` manifests for gang scheduling.

**Exercise hooks:**
- *(Medium)* Write a Karpenter `NodePool` manifest that targets `gpu-nvidia` instance types, sets a weight preference for spot, and caps at 32 nodes. Validate the manifest with `kubectl apply --dry-run=client`.
- *(Hard)* Configure a KAI Scheduler `Queue` with two teams: team-a gets 70% GPU quota, team-b gets 30%. Write both the `Queue` and `ResourceFlavor` manifests. Print the admitted workloads using `kubectl get queue team-a -o jsonpath='{.status}'`.

---

## Beat 6: Stretch

Multi-cluster scheduling with Karmada-based federation. Bin-packing vs. spread strategies for GPU nodes and when each matters. Spot instance interruption handling with checkpoint-based resume for training jobs.

**Exercise hooks:**
- *(Hard)* Write a script that queries the Karpenter `NodeClaim` API every 5 seconds, logs provisioning latency, and prints a p50/p95 summary after 20 samples. Output must be plain text to stdout.

---

## Learning Objectives

1. **Explain** why default Cluster Autoscaler introduces GPU provisioning latency and how Karpenter's event-driven loop reduces it.
2. **Configure** a Karpenter `NodePool` for GPU instance types with spot preferences and disruption controls.
3. **Implement** gang scheduling via `PodGroup` CRD to ensure all-or-nothing admission for distributed training jobs.
4. **Compare** default kube-scheduler bin-packing against KAI Scheduler's queue-based GPU-aware scheduling with priority and preemption.
5. **Calculate** the cost impact of provisioning delays on GPU utilization and relate it to inference/training unit economics.

---

## GTM Redirect Rules

- **Use It (Beat 4):** Direct redirect — GPU autoscaling controls per-inference cost and throughput capacity, which is the infrastructure layer underneath any GTM system doing real-time scoring or enrichment. Specifically: if you run inference endpoints for lead scoring or enrichment (Zone 2, enrichment pipelines), autoscaling is what keeps latency stable without over-provisioning.
- **Ship It (Beat 5):** Production manifests are infrastructure-agnostic. The redirect is: "these are the manifests that keep GPU inference endpoints from over-provisioning during low traffic — foundational for Zone 2 enrichment pipeline reliability."
- If the AI concept does not cleanly map to a GTM motion, the redirect is: "foundational for Zone 1" — no fabricated application.