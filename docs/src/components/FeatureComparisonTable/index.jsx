import React from "react";
import styles from "./FeatureComparisonTable.module.css";

const datasets = {
    ragas: {
      summary: [
        {
          feature: "RAG metrics",
          description: "The popular RAG metrics such as faithfulness",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Conversational metrics",
          description: "Evaluates LLM chatbot conversationals",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Agentic metrics",
          description: "Evaluates agentic workflows, tool use",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Safety LLM red teaming",
          description:
            "Metrics for LLM safety and security like bias, PII leakage",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-modal LLM evaluation",
          description: "Metrics involving image generations as well",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, research-backed metrics",
          description: "Custom metrics builder with research-backing",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, deterministic metrics",
          description: "Custom, LLM powered decision-based metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Open-source",
          description: "Open with nothing to hide",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LLM evaluation platform",
          description:
            "Testing reports, regression A|B testing, metric analysis, metric validation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM observability platform",
          description: "LLM tracing, monitoring, cost & latency tracking",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Enterprise-ready platform",
          description: "SSO, compliance, user roles & permissions, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Is Confident in their product",
          description: "Just kidding",
          deepeval: true,
          competitor: false,
        },
      ],
      metrics: [
        {
          feature: "RAG metrics",
          description: "The popular RAG metrics such as faithfulness",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Conversational metrics",
          description: "Evaluates LLM chatbot conversationals",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Agentic metrics",
          description: "Evaluates agentic workflows, tool use",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Red teaming metrics",
          description:
            "Metrics for LLM safety and security like bias, PII leakage",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-modal metrics",
          description: "Metrics involving image generations as well",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Use case specific metrics",
          description: "Summarization, JSON correctness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, research-backed metrics",
          description: "Custom metrics builder should have research-backing",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, deterministic metrics",
          description: "Custom, LLM powered decision-based metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Fully customizable metrics",
          description: "Use existing metric templates for full customization",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Explanability",
          description: "Metric provides reasons for all runs",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Run using any LLM judge",
          description: "Not vendor-locked into any framework for LLM providers",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "JSON-confineable",
          description:
            "Custom LLM judges can be forced to output valid JSON for metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Verbose debugging",
          description: "Debug LLM thinking processes during evaluation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Caching",
          description: "Optionally save metric scores to avoid re-computation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Cost tracking",
          description: "Track LLM judge token usage cost for each metric run",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Integrates with Confident AI",
          description: "Custom metrics or not, whether it can be on the cloud",
          deepeval: true,
          competitor: false,
        },
      ],
      synthesizer: [
        {
          feature: "Generate from documents",
          description: "Synthesize goldens that are grounded in documents",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Generate from ground truth",
          description: "Synthesize goldens that are grounded in context",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Generate free form goldens",
          description: "Synthesize goldens that are not grounded",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Quality filtering",
          description: "Remove goldens that do not meet the quality standards",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Non vendor-lockin",
          description: "No Langchain, LlamaIndex, etc. required",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Customize language",
          description:
            "Generate in français, español, deutsch, italiano, 日本語, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Customize output format",
          description: "Generate SQL, code, etc. not just simple QA",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Supports any LLMs",
          description: "Generate using any LLMs, with JSON confinement",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Save generations to Confident AI",
          description: "Not just generate, but bring it to your organization",
          deepeval: true,
          competitor: false,
        },
      ],
      redTeaming: [
        {
          feature: "Predefined vulnerabilities",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Attack simulation",
          description: "Simulate adversarial attacks to expose vulnerabilities",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Single-turn attack methods",
          description: "Prompt injection, ROT-13, leetspeak, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-turn attack methods",
          description: "Linear jailbreaking, tree jailbreaking, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Data privacy metrics",
          description: "PII leakage, prompt leakage, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Responsible AI metrics",
          description: "Bias, toxicity, fairness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Unauthorized access metrics",
          description: "RBAC, SSRF, shell injection, sql injection, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Brand image metrics",
          description: "Misinformation, IP infringement, robustness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Illegal risks metrics",
          description: "Illegal activity, graphic content, personal safety, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "OWASP Top 10 for LLMs",
          description: "Follows industry guidelines and standards",
          deepeval: true,
          competitor: false,
        },
      ],
      benchmarks: [
        {
          feature: "MMLU",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HellaSwag",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Big-Bench Hard",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "DROP",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "TruthfulQA",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HellaSwag",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
      ],
      integrations: [
        {
          feature: "Pytest",
          description: "First-class integration with Pytest for testing in CI/CD",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LangChain & LangGraph",
          description:
            "Run evals within the Lang ecosystem, or apps built with it",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LlamaIndex",
          description:
            "Run evals within the LlamaIndex ecosystem, or apps built with it",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Hugging Face",
          description: "Run evals during fine-tuning/training of models",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "ChromaDB",
          description: "Run evals on RAG pipelines built on Chroma",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Weaviate",
          description: "Run evals on RAG pipelines built on Weaviate",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Elastic",
          description: "Run evals on RAG pipelines built on Elastic",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "QDrant",
          description: "Run evals on RAG pipelines built on Qdrant",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "PGVector",
          description: "Run evals on RAG pipelines built on PGVector",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Langsmith",
          description: "Can be used within the Langsmith platform",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Helicone",
          description: "Can be used within the Helicone platform",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Confident AI",
          description: "Integrated with Confident AI",
          deepeval: true,
          competitor: false,
        },
      ],
      platform: [
        {
          feature: "Metric annotation",
          description: "Annotate the correctness of each metric",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Sharable testing reports",
          description:
            "Comprehensive reports that can be shared with stakeholders",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "A|B regression testing",
          description: "Determine any breaking changes before deployment",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Prompts and models experimentation",
          description: "Figure out which prompts and models work best",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Dataset editor",
          description: "Domain experts can edit datasets on the cloud",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Dataset revision history & backups",
          description: "Point in time recovery, edit history, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Metric score analysis",
          description:
            "Score distributions, mean, median, standard deviation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Metric validation",
          description:
            "False positives, false negatives, confusion matrices, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Prompt versioning",
          description: "Edit and manage prompts on the cloud instead of CSV",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Metrics on the cloud",
          description: "Run metrics on the platform instead of locally",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Trigger evals via HTTPs",
          description: "For users that are using (java/type)script",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Trigger evals without code",
          description: "For stakeholders that are non-technical",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Alerts and notifications",
          description:
            "Pings your slack, teams, discord, after each evaluation run.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM observability & tracing",
          description: "Monitor LLM interactions in production",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Online metrics in production",
          description: "Continuously monitor LLM performance",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Human feedback collection",
          description: "Collect feedback from internal team members or end users",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM guardrails",
          description: "Ultra-low latency guardrails in production",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM red teaming",
          description: "Managed LLM safety testing and attack curation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Self-hosting",
          description: "On-prem deployment so nothing leaves your data center",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "SSO",
          description: "Authenticate with your Idp of choice",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "User roles & permissions",
          description:
            "Custom roles, permissions, data segregation for different teams",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Transparent pricing",
          description: "Pricing should be available on the website",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HIPAA-ready",
          description: "For companies in the healthcare industry",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "SOCII certification",
          description: "For companies that need additional security compliance",
          deepeval: true,
          competitor: false,
        },
      ],
    },
    trulens: {
      metrics: [
        {
          feature: "RAG metrics",
          description: "The popular RAG metrics such as faithfulness",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Conversational metrics",
          description: "Evaluates LLM chatbot conversationals",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Agentic metrics",
          description: "Evaluates agentic workflows, tool use",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Red teaming metrics",
          description:
            "Metrics for LLM safety and security like bias, PII leakage",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-modal metrics",
          description: "Metrics involving image generations as well",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Use case specific metrics",
          description: "Summarization, JSON correctness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, research-backed metrics",
          description: "Custom metrics builder should have research-backing",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, deterministic metrics",
          description: "Custom, LLM powered decision-based metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Fully customizable metrics",
          description: "Use existing metric templates for full customization",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Explanability",
          description: "Metric provides reasons for all runs",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Run using any LLM judge",
          description: "Not vendor-locked into any framework for LLM providers",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "JSON-confineable",
          description:
            "Custom LLM judges can be forced to output valid JSON for metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Verbose debugging",
          description: "Debug LLM thinking processes during evaluation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Caching",
          description: "Optionally save metric scores to avoid re-computation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Cost tracking",
          description: "Track LLM judge token usage cost for each metric run",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Integrates with Confident AI",
          description: "Custom metrics or not, whether it can be on the cloud",
          deepeval: true,
          competitor: false,
        },
      ],
      synthesizer: [
        {
          feature: "Generate from documents",
          description: "Synthesize goldens that are grounded in documents",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Generate from ground truth",
          description: "Synthesize goldens that are grounded in context",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Generate free form goldens",
          description: "Synthesize goldens that are not grounded",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Quality filtering",
          description: "Remove goldens that do not meet the quality standards",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Non vendor-lockin",
          description: "No Langchain, LlamaIndex, etc. required",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Customize language",
          description:
            "Generate in français, español, deutsch, italiano, 日本語, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Customize output format",
          description: "Generate SQL, code, etc. not just simple QA",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Supports any LLMs",
          description: "Generate using any LLMs, with JSON confinement",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Save generations to Confident AI",
          description: "Not just generate, but bring it to your organization",
          deepeval: true,
          competitor: false,
        },
      ],
      redTeaming: [
        {
          feature: "Predefined vulnerabilities",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Attack simulation",
          description: "Simulate adversarial attacks to expose vulnerabilities",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Single-turn attack methods",
          description: "Prompt injection, ROT-13, leetspeak, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-turn attack methods",
          description: "Linear jailbreaking, tree jailbreaking, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Data privacy metrics",
          description: "PII leakage, prompt leakage, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Responsible AI metrics",
          description: "Bias, toxicity, fairness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Unauthorized access metrics",
          description: "RBAC, SSRF, shell injection, sql injection, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Brand image metrics",
          description: "Misinformation, IP infringement, robustness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Illegal risks metrics",
          description: "Illegal activity, graphic content, personal safety, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "OWASP Top 10 for LLMs",
          description: "Follows industry guidelines and standards",
          deepeval: true,
          competitor: false,
        },
      ],
      benchmarks: [
        {
          feature: "MMLU",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HellaSwag",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Big-Bench Hard",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "DROP",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "TruthfulQA",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HellaSwag",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
      ],
      integrations: [
        {
          feature: "Pytest",
          description: "First-class integration with Pytest for testing in CI/CD",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LangChain & LangGraph",
          description:
            "Run evals within the Lang ecosystem, or apps built with it",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LlamaIndex",
          description:
            "Run evals within the LlamaIndex ecosystem, or apps built with it",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Hugging Face",
          description: "Run evals during fine-tuning/training of models",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "ChromaDB",
          description: "Run evals on RAG pipelines built on Chroma",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Weaviate",
          description: "Run evals on RAG pipelines built on Weaviate",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Elastic",
          description: "Run evals on RAG pipelines built on Elastic",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "QDrant",
          description: "Run evals on RAG pipelines built on Qdrant",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "PGVector",
          description: "Run evals on RAG pipelines built on PGVector",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Snowflake",
          description: "Integrated with Snowflake logs",
          deepeval: false,
          competitor: true,
        },
        {
          feature: "Confident AI",
          description: "Integrated with Confident AI",
          deepeval: true,
          competitor: false,
        },
      ],
      platform: [
        {
          feature: "Sharable testing reports",
          description:
            "Comprehensive reports that can be shared with stakeholders",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "A|B regression testing",
          description: "Determine any breaking changes before deployment",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Prompts and models experimentation",
          description: "Figure out which prompts and models work best",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Dataset editor",
          description: "Domain experts can edit datasets on the cloud",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Dataset revision history & backups",
          description: "Point in time recovery, edit history, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Metric score analysis",
          description:
            "Score distributions, mean, median, standard deviation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Metric annotation",
          description: "Annotate the correctness of each metric",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Metric validation",
          description:
            "False positives, false negatives, confusion matrices, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Prompt versioning",
          description: "Edit and manage prompts on the cloud instead of CSV",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Metrics on the cloud",
          description: "Run metrics on the platform instead of locally",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Trigger evals via HTTPs",
          description: "For users that are using (java/type)script",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Trigger evals without code",
          description: "For stakeholders that are non-technical",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Alerts and notifications",
          description:
            "Pings your slack, teams, discord, after each evaluation run.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM observability & tracing",
          description: "Monitor LLM interactions in production",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Online metrics in production",
          description: "Continuously monitor LLM performance",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Human feedback collection",
          description: "Collect feedback from internal team members or end users",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LLM guardrails",
          description: "Ultra-low latency guardrails in production",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM red teaming",
          description: "Managed LLM safety testing and attack curation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Self-hosting",
          description: "On-prem deployment so nothing leaves your data center",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "SSO",
          description: "Authenticate with your Idp of choice",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "User roles & permissions",
          description:
            "Custom roles, permissions, data segregation for different teams",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Transparent pricing",
          description: "Pricing should be available on the website",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HIPAA-ready",
          description: "For companies in the healthcare industry",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "SOCII certification",
          description: "For companies that need additional security compliance",
          deepeval: true,
          competitor: false,
        },
      ],
    },
    arize: {
      summary: [
        {
          feature: "RAG metrics",
          description: "The popular RAG metrics such as faithfulness",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Conversational metrics",
          description: "Evaluates LLM chatbot conversationals",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Agentic metrics",
          description: "Evaluates agentic workflows, tool use",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Safety LLM red teaming",
          description:
            "Metrics for LLM safety and security like bias, PII leakage",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-modal LLM evaluation",
          description: "Metrics involving image generations as well",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, research-backed metrics",
          description: "Custom metrics builder with research-backing",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, deterministic metrics",
          description: "Custom, LLM powered decision-based metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Open-source",
          description: "Open with nothing to hide",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LLM evaluation platform",
          description:
            "Testing reports, regression A|B testing, metric analysis, metric validation",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "LLM observability platform",
          description: "LLM tracing, monitoring, cost & latency tracking",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Enterprise-ready platform",
          description: "SSO, compliance, user roles & permissions, etc.",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Is Confident in their product",
          description: "Just kidding",
          deepeval: true,
          competitor: false,
        },
      ],
      metrics: [
        {
          feature: "RAG metrics",
          description: "The popular RAG metrics such as faithfulness",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Conversational metrics",
          description: "Evaluates LLM chatbot conversationals",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Agentic metrics",
          description: "Evaluates agentic workflows, tool use",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Red teaming metrics",
          description:
            "Metrics for LLM safety and security like bias, PII leakage",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-modal metrics",
          description: "Metrics involving image generations as well",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Use case specific metrics",
          description: "Summarization, JSON correctness, etc.",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Custom, research-backed metrics",
          description: "Custom metrics builder should have research-backing",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, deterministic metrics",
          description: "Custom, LLM powered decision-based metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Fully customizable metrics",
          description: "Use existing metric templates for full customization",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Explanability",
          description: "Metric provides reasons for all runs",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Run using any LLM judge",
          description: "Not vendor-locked into any framework for LLM providers",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "JSON-confineable",
          description:
            "Custom LLM judges can be forced to output valid JSON for metrics",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Verbose debugging",
          description: "Debug LLM thinking processes during evaluation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Caching",
          description: "Optionally save metric scores to avoid re-computation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Cost tracking",
          description: "Track LLM judge token usage cost for each metric run",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Integrates with Confident AI",
          description: "Custom metrics or not, whether it can be on the cloud",
          deepeval: true,
          competitor: false,
        },
      ],
      synthesizer: [
        {
          feature: "Generate from documents",
          description: "Synthesize goldens that are grounded in documents",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Generate from ground truth",
          description: "Synthesize goldens that are grounded in context",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Generate free form goldens",
          description: "Synthesize goldens that are not grounded",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Quality filtering",
          description: "Remove goldens that do not meet the quality standards",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Non vendor-lockin",
          description: "No Langchain, LlamaIndex, etc. required",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Customize language",
          description:
            "Generate in français, español, deutsch, italiano, 日本語, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Customize output format",
          description: "Generate SQL, code, etc. not just simple QA",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Supports any LLMs",
          description: "Generate using any LLMs, with JSON confinement",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Save generations to Confident AI",
          description: "Not just generate, but bring it to your organization",
          deepeval: true,
          competitor: false,
        },
      ],
      redTeaming: [
        {
          feature: "Predefined vulnerabilities",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Attack simulation",
          description: "Simulate adversarial attacks to expose vulnerabilities",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Single-turn attack methods",
          description: "Prompt injection, ROT-13, leetspeak, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-turn attack methods",
          description: "Linear jailbreaking, tree jailbreaking, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Data privacy metrics",
          description: "PII leakage, prompt leakage, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Responsible AI metrics",
          description: "Bias, toxicity, fairness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Unauthorized access metrics",
          description: "RBAC, SSRF, shell injection, sql injection, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Brand image metrics",
          description: "Misinformation, IP infringement, robustness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Illegal risks metrics",
          description: "Illegal activity, graphic content, personal safety, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "OWASP Top 10 for LLMs",
          description: "Follows industry guidelines and standards",
          deepeval: true,
          competitor: false,
        },
      ],
      benchmarks: [
        {
          feature: "MMLU",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HellaSwag",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Big-Bench Hard",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "DROP",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "TruthfulQA",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HellaSwag",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
      ],
      integrations: [
        {
          feature: "Pytest",
          description: "First-class integration with Pytest for testing in CI/CD",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LangChain & LangGraph",
          description:
            "Run evals within the Lang ecosystem, or apps built with it",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LlamaIndex",
          description:
            "Run evals within the LlamaIndex ecosystem, or apps built with it",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Hugging Face",
          description: "Run evals during fine-tuning/training of models",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "ChromaDB",
          description: "Run evals on RAG pipelines built on Chroma",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Weaviate",
          description: "Run evals on RAG pipelines built on Weaviate",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Elastic",
          description: "Run evals on RAG pipelines built on Elastic",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "QDrant",
          description: "Run evals on RAG pipelines built on Qdrant",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "PGVector",
          description: "Run evals on RAG pipelines built on PGVector",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Langsmith",
          description: "Can be used within the Langsmith platform",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Helicone",
          description: "Can be used within the Helicone platform",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Confident AI",
          description: "Integrated with Confident AI",
          deepeval: true,
          competitor: false,
        },
      ],
      platform: [
        {
          feature: "Metric annotation",
          description: "Annotate the correctness of each metric",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Sharable testing reports",
          description:
            "Comprehensive reports that can be shared with stakeholders",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "A|B regression testing",
          description: "Determine any breaking changes before deployment",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Prompts and models experimentation",
          description: "Figure out which prompts and models work best",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Dataset editor",
          description: "Domain experts can edit datasets on the cloud",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Dataset revision history & backups",
          description: "Point in time recovery, edit history, etc.",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Metric score analysis",
          description:
            "Score distributions, mean, median, standard deviation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Metric validation",
          description:
            "False positives, false negatives, confusion matrices, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Prompt versioning",
          description: "Edit and manage prompts on the cloud instead of CSV",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Metrics on the cloud",
          description: "Run metrics on the platform instead of locally",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Trigger evals via HTTPs",
          description: "For users that are using (java/type)script",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Trigger evals without code",
          description: "For stakeholders that are non-technical",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Alerts and notifications",
          description:
            "Pings your slack, teams, discord, after each evaluation run.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM observability & tracing",
          description: "Monitor LLM interactions in production",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Online metrics in production",
          description: "Continuously monitor LLM performance",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Human feedback collection",
          description: "Collect feedback from internal team members or end users",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LLM guardrails",
          description: "Ultra-low latency guardrails in production",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM red teaming",
          description: "Managed LLM safety testing and attack curation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Self-hosting",
          description: "On-prem deployment so nothing leaves your data center",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "SSO",
          description: "Authenticate with your Idp of choice",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "User roles & permissions",
          description:
            "Custom roles, permissions, data segregation for different teams",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Transparent pricing",
          description: "Pricing should be available on the website",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "HIPAA-ready",
          description: "For companies in the healthcare industry",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "SOCII certification",
          description: "For companies that need additional security compliance",
          deepeval: true,
          competitor: true,
        },
      ],
    },
    langfuse: {
      metrics: [
        {
          feature: "RAG metrics",
          description: "The popular RAG metrics such as faithfulness",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Conversational metrics",
          description: "Evaluates LLM chatbot conversationals",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Agentic metrics",
          description: "Evaluates agentic workflows, tool use",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Red teaming metrics",
          description:
            "Metrics for LLM safety and security like bias, PII leakage",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-modal metrics",
          description: "Metrics involving image generations as well",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Use case specific metrics",
          description: "Summarization, JSON correctness, etc.",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Custom, research-backed metrics",
          description: "Custom metrics builder should have research-backing",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, deterministic metrics",
          description: "Custom, LLM powered decision-based metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Fully customizable metrics",
          description: "Use existing metric templates for full customization",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Explanability",
          description: "Metric provides reasons for all runs",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Run using any LLM judge",
          description: "Not vendor-locked into any framework for LLM providers",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "JSON-confineable",
          description:
            "Custom LLM judges can be forced to output valid JSON for metrics",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Verbose debugging",
          description: "Debug LLM thinking processes during evaluation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Caching",
          description: "Optionally save metric scores to avoid re-computation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Cost tracking",
          description: "Track LLM judge token usage cost for each metric run",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Integrates with Confident AI",
          description: "Custom metrics or not, whether it can be on the cloud",
          deepeval: true,
          competitor: false,
        },
      ],
      synthesizer: [
        {
          feature: "Generate from documents",
          description: "Synthesize goldens that are grounded in documents",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Generate from ground truth",
          description: "Synthesize goldens that are grounded in context",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Generate free form goldens",
          description: "Synthesize goldens that are not grounded",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Quality filtering",
          description: "Remove goldens that do not meet the quality standards",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Non vendor-lockin",
          description: "No Langchain, LlamaIndex, etc. required",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Customize language",
          description:
            "Generate in français, español, deutsch, italiano, 日本語, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Customize output format",
          description: "Generate SQL, code, etc. not just simple QA",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Supports any LLMs",
          description: "Generate using any LLMs, with JSON confinement",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Save generations to Confident AI",
          description: "Not just generate, but bring it to your organization",
          deepeval: true,
          competitor: false,
        },
      ],
      redTeaming: [
        {
          feature: "Predefined vulnerabilities",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Attack simulation",
          description: "Simulate adversarial attacks to expose vulnerabilities",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Single-turn attack methods",
          description: "Prompt injection, ROT-13, leetspeak, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-turn attack methods",
          description: "Linear jailbreaking, tree jailbreaking, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Data privacy metrics",
          description: "PII leakage, prompt leakage, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Responsible AI metrics",
          description: "Bias, toxicity, fairness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Unauthorized access metrics",
          description: "RBAC, SSRF, shell injection, sql injection, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Brand image metrics",
          description: "Misinformation, IP infringement, robustness, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Illegal risks metrics",
          description: "Illegal activity, graphic content, personal safety, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "OWASP Top 10 for LLMs",
          description: "Follows industry guidelines and standards",
          deepeval: true,
          competitor: false,
        },
      ],
      benchmarks: [
        {
          feature: "MMLU",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HellaSwag",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Big-Bench Hard",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "DROP",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "TruthfulQA",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "HellaSwag",
          description:
            "Vulnerabilities such as bias, toxicity, misinformation, etc.",
          deepeval: true,
          competitor: false,
        },
      ],
      integrations: [
        {
          feature: "Pytest",
          description: "First-class integration with Pytest for testing in CI/CD",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LangChain & LangGraph",
          description:
            "Run evals within the Lang ecosystem, or apps built with it",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LlamaIndex",
          description:
            "Run evals within the LlamaIndex ecosystem, or apps built with it",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Hugging Face",
          description: "Run evals during fine-tuning/training of models",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "ChromaDB",
          description: "Run evals on RAG pipelines built on Chroma",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Weaviate",
          description: "Run evals on RAG pipelines built on Weaviate",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Elastic",
          description: "Run evals on RAG pipelines built on Elastic",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "QDrant",
          description: "Run evals on RAG pipelines built on Qdrant",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "PGVector",
          description: "Run evals on RAG pipelines built on PGVector",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Langsmith",
          description: "Can be used within the Langsmith platform",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Helicone",
          description: "Can be used within the Helicone platform",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Confident AI",
          description: "Integrated with Confident AI",
          deepeval: true,
          competitor: false,
        },
      ],
      platform: [
        {
          feature: "Metric annotation",
          description: "Annotate the correctness of each metric",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Sharable testing reports",
          description:
            "Comprehensive reports that can be shared with stakeholders",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "A|B regression testing",
          description: "Determine any breaking changes before deployment",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Prompts and models experimentation",
          description: "Figure out which prompts and models work best",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Dataset editor",
          description: "Domain experts can edit datasets on the cloud",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Dataset revision history & backups",
          description: "Point in time recovery, edit history, etc.",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Metric score analysis",
          description:
            "Score distributions, mean, median, standard deviation, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Metric validation",
          description:
            "False positives, false negatives, confusion matrices, etc.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Prompt versioning",
          description: "Edit and manage prompts on the cloud instead of CSV",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Metrics on the cloud",
          description: "Run metrics on the platform instead of locally",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Trigger evals via HTTPs",
          description: "For users that are using (java/type)script",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Trigger evals without code",
          description: "For stakeholders that are non-technical",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Alerts and notifications",
          description:
            "Pings your slack, teams, discord, after each evaluation run.",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM observability & tracing",
          description: "Monitor LLM interactions in production",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Online metrics in production",
          description: "Continuously monitor LLM performance",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Human feedback collection",
          description: "Collect feedback from internal team members or end users",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LLM guardrails",
          description: "Ultra-low latency guardrails in production",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM red teaming",
          description: "Managed LLM safety testing and attack curation",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Self-hosting",
          description: "On-prem deployment so nothing leaves your data center",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "SSO",
          description: "Authenticate with your Idp of choice",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "User roles & permissions",
          description:
            "Custom roles, permissions, data segregation for different teams",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Transparent pricing",
          description: "Pricing should be available on the website",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "HIPAA-ready",
          description: "For companies in the healthcare industry",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "SOCII certification",
          description: "For companies that need additional security compliance",
          deepeval: true,
          competitor: true,
        },
      ],
    },
    braintrust: {
      summary: [
        {
          feature: "RAG metrics",
          description: "The popular RAG metrics such as faithfulness",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Conversational metrics",
          description: "Evaluates LLM chatbot conversationals",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Agentic metrics",
          description: "Evaluates agentic workflows, tool use",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Safety LLM red teaming",
          description:
            "Metrics for LLM safety and security like bias, PII leakage",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Multi-modal LLM evaluation",
          description: "Metrics involving image generations as well",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, research-backed metrics",
          description: "Custom metrics builder with research-backing",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, deterministic metrics",
          description: "Custom, LLM powered decision-based metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Open-source",
          description: "Open with nothing to hide",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "LLM evaluation platform",
          description:
            "Testing reports, regression A|B testing, metric analysis, metric validation",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LLM observability platform",
          description: "LLM tracing, monitoring, cost & latency tracking",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Enterprise-ready platform",
          description: "SSO, compliance, user roles & permissions, etc.",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Is Confident in their product",
          description: "Just kidding",
          deepeval: true,
          competitor: false,
        },
      ],
    },
    promptfoo: {
      summary: [
        {
          feature: "RAG metrics",
          description: "The popular RAG metrics such as faithfulness",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Conversational metrics",
          description: "Evaluates LLM chatbot conversationals",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Agentic metrics",
          description: "Evaluates agentic workflows, tool use",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Safety LLM red teaming",
          description:
            "Metrics for LLM safety and security like bias, PII leakage",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Multi-modal LLM evaluation",
          description: "Metrics involving image generations as well",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Custom, research-backed metrics",
          description: "Custom metrics builder with research-backing",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "Custom, deterministic metrics",
          description: "Custom, LLM powered decision-based metrics",
          deepeval: true,
          competitor: false,
        },
        {
          feature: "Open-source",
          description: "Open with nothing to hide",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LLM evaluation platform",
          description:
            "Testing reports, regression A|B testing, metric analysis, metric validation",
          deepeval: true,
          competitor: true,
        },
        {
          feature: "LLM observability platform",
          description: "LLM tracing, monitoring, cost & latency tracking",
          deepeval: true,
          competitor: "Limited",
        },
        {
          feature: "Enterprise-ready platform",
          description: "SSO, compliance, user roles & permissions, etc.",
          deepeval: true,
          competitor: "Half-way there",
        },
        {
          feature: "Is Confident in their product",
          description: "Just kidding",
          deepeval: true,
          competitor: false,
        },
      ],
    },
  };
  
export default function FeatureComparisonTable({ type, competitor }) {
  const [topKey, subKey] = type.split("::");
  const data = datasets?.[topKey]?.[subKey] || [];

  const renderValue = (value) => {
    if (typeof value === "string") {
      return <span className={styles.cellText}>{value}</span>;
    }

    return value ? (
      <img alt="yes" src="/icons/tick.svg" className={styles.tick} />
    ) : (
      <img alt="no" src="/icons/cross.svg" className={styles.cross} />
    );
  };

  return (
    <div className={styles.tableContainer}>
      <div className={styles.featureTable}>
        <div className={styles.featureHeader}>
          <div className={styles.featureCell}></div>
          <div className={styles.centeredCell}>DeepEval</div>
          <div className={styles.centeredCell}>{competitor}</div>
        </div>
        <div>
          {data.map((item, idx) => (
            <div key={idx} className={styles.featureRow}>
              <div className={styles.featureCell}>
                <span className={styles.featureTitle}>{item.feature}</span>
                <div className={styles.featureDescription}>
                  {item.description}
                </div>
              </div>
              <div className={styles.centeredCell}>
                {renderValue(item.deepeval)}
              </div>
              <div className={styles.centeredCell}>
                {renderValue(item.competitor)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 