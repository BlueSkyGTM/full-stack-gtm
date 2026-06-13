# Regulatory Frameworks — EU, US, UK, Korea

## Hook

Data privacy and AI governance regulations now dictate what you can collect, how you can process it, and when you must explain automated decisions. A GTM stack that violates the EU AI Act or Korea's PIPA isn't just non-compliant — it's a liability that blocks market entry.

## Concept

Four regulatory models, four enforcement philosophies. The EU operates on rights-based precaution with extraterritorial reach. The US fragments across sector-specific and state-level rules. The UK pivoted toward "pro-innovation" principles post-Brexit. Korea harmonized with GDPR structurally but added consent requirements stricter than either EU or US frameworks. This beat covers the classification schemes, extraterritoriality triggers, and enforcement mechanisms for each.

## Apply

Map a hypothetical GTM data pipeline — collecting prospect data, enriching via third-party tools, scoring with an ML model, and sending automated outreach — against each jurisdiction's requirements. Identify where the same pipeline is legal in one region and unlawful in another. Exercise hooks: Easy — classify which regulatory framework applies given a company's target market and data practices. Medium — map a GTM data flow against GDPR Article 6 lawful bases and identify gaps. Hard — write a compliance decision function that takes a data processing description and returns required actions per jurisdiction.

## Use It

GTM Cluster: Zone 01 — Data Foundation and Enrichment. Regulatory compliance gates every data enrichment waterfall and prospecting workflow. When you pull contact data through Clay or any enrichment tool, the lawful basis for processing and the territorial scope of the regulation determine whether that data can be used at all. This is the compliance layer that sits underneath every GTM data operation.

## Ship It

Build a jurisdiction-aware data handling config that tags each prospect record with applicable regulatory constraints based on their location, your company's establishment, and the processing activity. Output a per-record compliance label. Exercise hooks: Easy — write a function that returns applicable regulations given a prospect country code. Medium — build a config file that defines permitted data fields per jurisdiction and a validator that flags violations. Hard — implement a compliance gate that blocks GTM workflow execution when a record's data processing would violate the most restrictive applicable framework, with an audit log of the decision.

## Evaluate

Assess a provided GTM data pipeline design against all four regulatory frameworks. Identify specific articles, sections, or rules violated. Propose remediations. Exercise hooks: Easy — multiple-choice identification of which regulation prohibits a described data practice. Medium — given a data enrichment workflow, list all regulatory violations across jurisdictions with specific article citations. Hard — redesign a non-compliant GTM pipeline to operate legally across all four jurisdictions simultaneously, justifying each design choice with regulatory text.