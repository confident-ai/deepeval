AgentLoopDetectionMetric MVP

1. Parse tool spans from _trace_dict
2. Create fingerprint:
   (tool_name, serialized_args)
3. Count duplicate consecutive calls
4. Score:
   1.0 = no loops
   0.5 = mild repetition
   0.0 = severe repetition
5. Reason:
   "Tool X called 4 times with identical arguments"
6. Add synthetic trace unit tests
7. Push update to PR #2782
