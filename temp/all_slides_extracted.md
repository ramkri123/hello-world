# PowerPoint Content Extract - All Slides
**Source:** Zero-trust Sovereign AI.pptx
**Total Slides:** 4


## Slide 1

Privacy-Preserving AI/ML Model Aggregation/Comparison for sovereign clouds


## Slide 2

Sovereign cloud organizations can compare AI models or aggregate AI model results without revealing raw data or model details.

Definition, Use cases and business value


## Slide 3

Federated learning challenges for use case realization

Sensitive model weights are exposed to central coordinator – for example Swift for Consortium Fraud & Risk Scoring

Using model weights, the original training data could be extracted – membership inference attack


## Slide 4

Consortium participant (aka collaborator)

Example privacy preserving consortium architecture

Local model training and inferencing

Bank on-prem 
-- Local fraud detection model; training data and model never leave premise
-- Provide proof of residency/geolocation to consortium aggregator/comparator

Local model training and inferencing

Local model training and inferencing

Secure Enclave (TEE)

Consortium aggregator/ comparator

Inference input reception and distribution
Inference output comparison/aggregation

Consortium participant (aka collaborator)

Consortium participant (aka collaborator)

Banking Consortiums (e.g., SWIFT‐backed initiatives)
-- Privacy preserving consortium fraud & risk scoring
-- Provide proof of residency/geolocation to consortium participant

Persistent network connection (only outbound connection from on-prem; no inbound on-prem port exposure)

AI Agent

Input: 
- High value bank transaction

Output: - Aggregated metric: average risk
- Comparison metric: Consensus vs. divergence rate

